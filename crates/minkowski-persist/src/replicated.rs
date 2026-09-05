//! Source membership, process-local follower sessions, and conservative retention.

use std::collections::BTreeMap;
use std::sync::Arc;

use minkowski::{Access, EnumChangeSet, Transact, TransactError, Tx, World, WorldMismatch};
use parking_lot::Mutex;

use crate::{
    Durable, Fetch, FetchResponse, JournaledFollower, ReplicationPump, TransportError, WalError,
    WalFrameRange, WalRangeLimits,
};

#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    #[error("member {0:?} is not configured")]
    UnknownMember(String),
    #[error("member {0:?} is already configured")]
    DuplicateMember(String),
    #[error("follower belongs to another source history")]
    HistoryMismatch,
    #[error("poisoned follower must recover before joining")]
    PoisonedFollower,
    #[error("follower prefix {applied} exceeds published source tail {published}")]
    Ahead { applied: u64, published: u64 },
}

#[derive(Default)]
struct Member {
    session: Option<Arc<()>>,
    consumed_seq: u64,
}

fn session_member<'a>(
    members: &'a mut BTreeMap<String, Member>,
    member: &str,
    session: &Arc<()>,
) -> Result<&'a mut Member, TransportError> {
    members
        .get_mut(member)
        .filter(|entry| {
            entry
                .session
                .as_ref()
                .is_some_and(|current| Arc::ptr_eq(current, session))
        })
        .ok_or(TransportError::RejoinRequired)
}

/// Informational retention policy, not authorization to delete WAL files.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RetentionPlan {
    pub leader_replay_floor: u64,
    pub follower_floor: Option<u64>,
    pub candidate_cutoff: u64,
    /// Currently zero: the original prefix is required to reconstruct fences.
    pub deletion_cutoff: u64,
}

/// Durable source with explicitly configured membership. Restart must supply
/// the complete authoritative membership again; every member starts at zero
/// until its journaled follower rejoins. Disconnect never removes a member.
///
/// Sessions are private in-process capabilities, not serialized credentials.
/// Each join replaces the capability, so an old client cannot update a new
/// session, even after removal/re-add or construction of another source owner.
pub struct Replicated<S: Transact> {
    durable: Durable<S>,
    history: [u8; 16],
    members: Mutex<BTreeMap<String, Member>>,
}

impl<S: Transact> Replicated<S> {
    /// Wrap the source before serving followers or running transactions.
    /// `history` identifies the configured authoritative WAL history; change it
    /// when replacing that history. It is not a network authentication token.
    pub fn new(
        durable: Durable<S>,
        history: [u8; 16],
        members: impl IntoIterator<Item = String>,
    ) -> Result<Self, SessionError> {
        let mut configured = BTreeMap::new();
        for member in members {
            if configured
                .insert(member.clone(), Member::default())
                .is_some()
            {
                return Err(SessionError::DuplicateMember(member));
            }
        }
        durable.pin_replication_prefix();
        Ok(Self {
            durable,
            history,
            members: Mutex::new(configured),
        })
    }

    /// Explicit membership addition. A member with no active session pins zero.
    pub fn add_member(&self, member: String) -> Result<(), SessionError> {
        let mut members = self.members.lock();
        if members.contains_key(&member) {
            return Err(SessionError::DuplicateMember(member));
        }
        members.insert(member, Member::default());
        Ok(())
    }

    /// Explicit policy removal, invalidating all of this member's old clients.
    pub fn remove_member(&self, member: &str) -> Result<(), SessionError> {
        self.members
            .lock()
            .remove(member)
            .ok_or_else(|| SessionError::UnknownMember(member.to_owned()))?;
        Ok(())
    }

    /// Bind a new session to this exact owned follower and its reconstructed
    /// progress. A failed join leaves any existing session unchanged. The
    /// consumed follower handle is dropped on error; its journal remains intact.
    pub fn join(
        &self,
        member: &str,
        follower: JournaledFollower,
    ) -> Result<ReplicationPump<SessionFetch<'_, S>>, SessionError> {
        if follower.source_history() != self.history {
            return Err(SessionError::HistoryMismatch);
        }
        if follower.is_poisoned() {
            return Err(SessionError::PoisonedFollower);
        }
        let applied = follower.applied_seq();
        let published = self.durable.durable_seq();
        if applied > published {
            return Err(SessionError::Ahead { applied, published });
        }
        let mut members = self.members.lock();
        let entry = members
            .get_mut(member)
            .ok_or_else(|| SessionError::UnknownMember(member.to_owned()))?;
        let session = Arc::new(());
        entry.session = Some(Arc::clone(&session));
        entry.consumed_seq = applied;
        Ok(ReplicationPump::new(
            follower,
            SessionFetch {
                source: self,
                member: member.to_owned(),
                session,
            },
        ))
    }

    /// Last valid request report (or joined baseline), never a response's end.
    pub fn member_progress(&self, member: &str) -> Option<u64> {
        self.members.lock().get(member).map(|m| m.consumed_seq)
    }

    /// Calculate a conservative proposal. Supply zero without a verified LSM
    /// baseline; otherwise use its recovery replay floor, never checkpoint
    /// `flush_seq`. This input is not validated baseline/deletion authority.
    pub fn retention_plan(&self, leader_replay_floor: u64) -> RetentionPlan {
        let published = self.durable.durable_seq();
        let members = self.members.lock();
        let follower_floor = members.values().map(|m| m.consumed_seq).min();
        let candidate_cutoff = leader_replay_floor
            .min(published)
            .min(follower_floor.unwrap_or(u64::MAX));
        RetentionPlan {
            leader_replay_floor,
            follower_floor,
            candidate_cutoff,
            // ponytail: retain the prefix until durable fence summaries and
            // verified baseline ownership can authorize actual reclamation.
            deletion_cutoff: 0,
        }
    }

    fn fetch_session(
        &self,
        member: &str,
        session: &Arc<()>,
        from_seq: u64,
        limits: WalRangeLimits,
    ) -> Result<FetchResponse, TransportError> {
        self.fetch_session_with(member, session, from_seq, || {
            self.durable.records_from(from_seq, limits)
        })
    }

    fn fetch_session_with(
        &self,
        member: &str,
        session: &Arc<()>,
        from_seq: u64,
        read: impl FnOnce() -> Result<WalFrameRange, WalError>,
    ) -> Result<FetchResponse, TransportError> {
        // Do not hold membership while waiting for the WAL. Check again after
        // copying: rejoin/removal may revoke this capability during the read.
        session_member(&mut self.members.lock(), member, session)?;
        let range = read()?;
        let mut members = self.members.lock();
        let entry = session_member(&mut members, member, session)?;
        entry.consumed_seq = entry.consumed_seq.max(from_seq);
        Ok(FetchResponse {
            history: self.history,
            range,
        })
    }
}

/// Client issued only by `Replicated::join`, for one member/session/source.
/// A raw request position still relies on the caller's applied-progress
/// discipline; the pump supplies it directly from its healthy journaled follower.
pub struct SessionFetch<'a, S: Transact> {
    source: &'a Replicated<S>,
    member: String,
    session: Arc<()>,
}

impl<S: Transact> Fetch for SessionFetch<'_, S> {
    fn fetch(
        &mut self,
        from_seq: u64,
        limits: WalRangeLimits,
    ) -> Result<FetchResponse, TransportError> {
        self.source
            .fetch_session(&self.member, &self.session, from_seq, limits)
    }
}

impl<S: Transact> Transact for Replicated<S> {
    fn begin(&self, world: &mut World, access: &Access) -> Result<Tx<'_>, WorldMismatch> {
        self.durable.begin(world, access)
    }

    fn try_commit(
        &self,
        tx: &mut Tx<'_>,
        world: &mut World,
    ) -> Result<EnumChangeSet, TransactError> {
        self.durable.try_commit(tx, world)
    }

    fn max_retries(&self) -> usize {
        self.durable.max_retries()
    }

    fn transact<R>(
        &self,
        world: &mut World,
        access: &Access,
        f: impl FnMut(&mut Tx<'_>, &mut World) -> R,
    ) -> Result<R, TransactError> {
        self.durable.transact(world, access, f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CodecRegistry, Wal, WalConfig, recover_world};
    use minkowski::Optimistic;

    #[test]
    fn session_restart_resets_members_and_rejects_old_capability() {
        let dir = tempfile::tempdir().unwrap();
        let source_path = dir.path().join("source");
        let follower_path = dir.path().join("follower");
        let history = [7; 16];
        let limits = WalRangeLimits {
            max_records: 1,
            max_bytes: 65536,
            max_control_frames: 64,
        };
        let codecs = || {
            let mut world = World::builder().memory_budget(1024 * 1024).build().unwrap();
            let mut codecs = CodecRegistry::new();
            codecs.register_as::<u32>("value", &mut world).unwrap();
            codecs
        };
        let mut world = World::builder().memory_budget(1024 * 1024).build().unwrap();
        world.register_component::<u32>();
        let wal = Wal::create(&source_path, &codecs(), WalConfig::default()).unwrap();
        let durable = Durable::new(Optimistic::new(&world), wal, codecs());
        let members = || ["a".to_owned(), "b".to_owned()];
        let source = Replicated::new(durable, history, members()).unwrap();
        let access = Access::of::<(&mut u32,)>(&mut world);
        source
            .transact(&mut world, &access, |tx, world| {
                tx.spawn(world, (42u32,));
            })
            .unwrap();
        let follower =
            JournaledFollower::create(&follower_path, history, codecs(), limits).unwrap();
        let mut pump = source.join("a", follower).unwrap();
        pump.pump_once().unwrap();
        pump.pump_once().unwrap();
        assert_eq!(source.member_progress("a"), Some(1));
        let old_session = Arc::clone(&pump.fetch().session);
        drop(pump);
        drop(source);
        drop(world);

        let mut wal = Wal::open(&source_path, &codecs(), WalConfig::default()).unwrap();
        let world = recover_world(
            &dir.path().join("lsm"),
            &dir.path().join("manifest.log"),
            &mut wal,
            &codecs(),
        )
        .unwrap();
        let source = Replicated::new(
            Durable::new(Optimistic::new(&world), wal, codecs()),
            history,
            members(),
        )
        .unwrap();
        assert_eq!(source.member_progress("a"), Some(0));
        assert_eq!(source.member_progress("b"), Some(0));
        assert_eq!(source.retention_plan(1).candidate_cutoff, 0);
        assert!(matches!(
            source.fetch_session("a", &old_session, 1, limits),
            Err(TransportError::RejoinRequired)
        ));

        let follower = JournaledFollower::open(&follower_path, history, codecs(), limits).unwrap();
        let mut pump = source.join("a", follower).unwrap();
        assert_eq!(source.member_progress("a"), Some(1));
        assert_eq!(pump.pump_once().unwrap(), 1);
        assert!(matches!(
            source.fetch_session("a", &old_session, 1, limits),
            Err(TransportError::RejoinRequired)
        ));
        assert_eq!(source.retention_plan(1).candidate_cutoff, 0); // b remains configured
        source.remove_member("b").unwrap();
        assert_eq!(source.retention_plan(1).candidate_cutoff, 1);
        assert_eq!(source.retention_plan(1).deletion_cutoff, 0);

        // A current request can become stale while its detached range is being
        // copied. Replace its session inside the read boundary deterministically.
        let current_session = Arc::clone(&pump.fetch().session);
        let result = source.fetch_session_with("a", &current_session, 1, || {
            let range = source.durable.records_from(1, limits)?;
            let replacement = JournaledFollower::create(
                &dir.path().join("replacement"),
                history,
                codecs(),
                limits,
            )
            .unwrap();
            drop(source.join("a", replacement).unwrap());
            Ok(range)
        });
        assert!(matches!(result, Err(TransportError::RejoinRequired)));
        assert_eq!(source.member_progress("a"), Some(0));
    }
}
