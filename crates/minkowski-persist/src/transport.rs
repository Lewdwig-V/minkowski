//! Caller-driven pull replication. Transport never runs on the commit path.
//!
//! ```no_run
//! use minkowski::Transact;
//! use minkowski_persist::{Durable, JournaledFollower, LoopbackFetch, ReplicationPump};
//!
//! fn follow_once<S: Transact>(source: &Durable<S>, follower: JournaledFollower, history: [u8; 16]) {
//!     let mut pump = ReplicationPump::new(follower, LoopbackFetch::new(source, history));
//!     let applied_prefix = pump.pump_once().expect("fetch and durable ingest");
//!     // Retain the pump and call again to report this prefix and fetch more.
//! }
//! ```

use minkowski::Transact;

use crate::{Durable, IngestError, JournaledFollower, WalError, WalFrameRange, WalRangeLimits};

/// Detached source response. Adapters must bind `history` to the configured
/// authoritative source; this identifier is not authentication or a session.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FetchResponse {
    pub history: [u8; 16],
    pub range: WalFrameRange,
}

#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    #[error("request or response lost; retry the same applied position")]
    Lost,
    #[error("link down; reconnect before retrying the same applied position")]
    Down,
    #[error("source requires rejoin; install verified state before resuming")]
    RejoinRequired,
    #[error(transparent)]
    Source(#[from] WalError),
}

/// A client bound to one configured source. Fetch reports an exclusive applied
/// prefix and asks for the next range; it must not invent follower progress.
/// Implementations release source locks before delaying or sending a response.
/// Network adapters must bound incoming allocations before constructing it.
pub trait Fetch {
    fn fetch(
        &mut self,
        from_seq: u64,
        limits: WalRangeLimits,
    ) -> Result<FetchResponse, TransportError>;
}

impl<F> Fetch for F
where
    F: FnMut(u64, WalRangeLimits) -> Result<FetchResponse, TransportError>,
{
    fn fetch(
        &mut self,
        from_seq: u64,
        limits: WalRangeLimits,
    ) -> Result<FetchResponse, TransportError> {
        self(from_seq, limits)
    }
}

/// Local adapter over the same durable reader a source server would call.
pub struct LoopbackFetch<'a, S: Transact> {
    source: &'a Durable<S>,
    history: [u8; 16],
}

impl<'a, S: Transact> LoopbackFetch<'a, S> {
    /// `history` must remain stable for this source history, including restart.
    /// Replacing/reusing the log history requires a new identity and rejoin.
    pub fn new(source: &'a Durable<S>, history: [u8; 16]) -> Self {
        Self { source, history }
    }
}

impl<S: Transact> Fetch for LoopbackFetch<'_, S> {
    fn fetch(
        &mut self,
        from_seq: u64,
        limits: WalRangeLimits,
    ) -> Result<FetchResponse, TransportError> {
        Ok(FetchResponse {
            history: self.history,
            range: self.source.records_from(from_seq, limits)?,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FetchRequest {
    pub from_seq: u64,
    pub limits: WalRangeLimits,
}

/// Request transcript for tests and diagnostics, including failed requests.
/// Call `take_requests` to drain it during long-running use.
pub struct RecordingFetch<F> {
    inner: F,
    // ponytail: retain O(requests) metadata; drain with take_requests for long runs.
    requests: Vec<FetchRequest>,
}

impl<F> RecordingFetch<F> {
    pub fn new(inner: F) -> Self {
        Self {
            inner,
            requests: Vec::new(),
        }
    }

    pub fn requests(&self) -> &[FetchRequest] {
        &self.requests
    }

    pub fn take_requests(&mut self) -> Vec<FetchRequest> {
        std::mem::take(&mut self.requests)
    }
}

impl<F: Fetch> Fetch for RecordingFetch<F> {
    fn fetch(
        &mut self,
        from_seq: u64,
        limits: WalRangeLimits,
    ) -> Result<FetchResponse, TransportError> {
        self.requests.push(FetchRequest { from_seq, limits });
        self.inner.fetch(from_seq, limits)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum PumpError {
    #[error("pump stopped; repair or rejoin before creating a new pump")]
    Stopped,
    #[error(transparent)]
    Transport(#[from] TransportError),
    #[error(transparent)]
    Ingest(#[from] IngestError),
}

/// Owns one journaled follower and one fetch client. `&mut self` serializes
/// requests; there is no worker thread, timer, or independently advancing cursor.
/// The caller handles backoff/reconnect for `Lost` and `Down` errors.
pub struct ReplicationPump<F> {
    follower: JournaledFollower,
    fetch: F,
    stopped: bool,
}

impl<F: Fetch> ReplicationPump<F> {
    /// Use `JournaledFollower::open` before construction when restarting.
    pub fn new(follower: JournaledFollower, fetch: F) -> Self {
        Self {
            follower,
            fetch,
            stopped: false,
        }
    }

    /// Report the applied prefix, fetch once, journal and apply the response.
    /// Returns the exclusive applied prefix; an unchanged result does not prove
    /// catch-up (the response may be a duplicate). The next call reports it,
    /// including when that next response contains only control context.
    /// Terminal errors stop all further fetches, including after partial apply.
    pub fn pump_once(&mut self) -> Result<u64, PumpError> {
        if self.stopped || self.follower.is_poisoned() {
            return Err(PumpError::Stopped);
        }
        let response = match self
            .fetch
            .fetch(self.follower.applied_seq(), self.follower.range_limits())
        {
            Ok(response) => response,
            Err(error) => {
                self.stopped = !matches!(error, TransportError::Lost | TransportError::Down);
                return Err(error.into());
            }
        };
        match self
            .follower
            .ingest_frames(response.history, &response.range)
        {
            Ok(applied) => Ok(applied),
            Err(error) => {
                self.stopped = true;
                Err(error.into())
            }
        }
    }

    pub fn follower(&self) -> &JournaledFollower {
        &self.follower
    }

    pub fn fetch(&self) -> &F {
        &self.fetch
    }

    /// Recover ownership for reconnect or shutdown. A poisoned follower still
    /// requires recovery; putting it into another pump cannot resume requests.
    pub fn into_parts(self) -> (JournaledFollower, F) {
        (self.follower, self.fetch)
    }
}
