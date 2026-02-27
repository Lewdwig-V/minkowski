use crate::bundle::Bundle;
use crate::component::Component;
use crate::entity::Entity;
use crate::world::World;

/// Deferred mutation buffer. Records commands during iteration,
/// applies them all at once to &mut World.
pub struct CommandBuffer {
    commands: Vec<Box<dyn FnOnce(&mut World) + Send>>,
}

impl CommandBuffer {
    pub fn new() -> Self {
        Self { commands: Vec::new() }
    }

    pub fn spawn<B: Bundle>(&mut self, bundle: B) {
        self.commands.push(Box::new(move |world| {
            world.spawn(bundle);
        }));
    }

    pub fn despawn(&mut self, entity: Entity) {
        self.commands.push(Box::new(move |world| {
            world.despawn(entity);
        }));
    }

    pub fn insert<T: Component>(&mut self, entity: Entity, component: T) {
        self.commands.push(Box::new(move |world| {
            world.insert(entity, component);
        }));
    }

    pub fn remove<T: Component>(&mut self, entity: Entity) {
        self.commands.push(Box::new(move |world| {
            world.remove::<T>(entity);
        }));
    }

    pub fn apply(self, world: &mut World) {
        for command in self.commands {
            command(world);
        }
    }

    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }
}

impl Default for CommandBuffer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::World;

    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Pos { x: f32, y: f32 }
    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Vel { dx: f32, dy: f32 }

    #[test]
    fn command_spawn() {
        let mut world = World::new();
        let mut cmds = CommandBuffer::new();
        cmds.spawn((Pos { x: 1.0, y: 2.0 },));
        cmds.spawn((Pos { x: 3.0, y: 4.0 },));
        cmds.apply(&mut world);

        let count = world.query::<&Pos>().count();
        assert_eq!(count, 2);
    }

    #[test]
    fn command_despawn() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 0.0 },));
        let mut cmds = CommandBuffer::new();
        cmds.despawn(e);
        cmds.apply(&mut world);

        assert!(!world.is_alive(e));
    }

    #[test]
    fn command_insert() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 0.0 },));
        let mut cmds = CommandBuffer::new();
        cmds.insert(e, Vel { dx: 5.0, dy: 0.0 });
        cmds.apply(&mut world);

        assert_eq!(world.get::<Vel>(e), Some(&Vel { dx: 5.0, dy: 0.0 }));
    }

    #[test]
    fn command_remove() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 0.0 }, Vel { dx: 5.0, dy: 0.0 }));
        let mut cmds = CommandBuffer::new();
        cmds.remove::<Vel>(e);
        cmds.apply(&mut world);

        assert_eq!(world.get::<Vel>(e), None);
        assert_eq!(world.get::<Pos>(e), Some(&Pos { x: 1.0, y: 0.0 }));
    }

    #[test]
    fn commands_during_iteration() {
        let mut world = World::new();
        for i in 0..5 {
            world.spawn((Pos { x: i as f32, y: 0.0 },));
        }

        let mut cmds = CommandBuffer::new();
        for (entity, pos) in world.query::<(crate::entity::Entity, &Pos)>() {
            if pos.x > 2.0 {
                cmds.despawn(entity);
            }
        }
        cmds.apply(&mut world);

        let count = world.query::<&Pos>().count();
        assert_eq!(count, 3); // 0.0, 1.0, 2.0 remain
    }
}
