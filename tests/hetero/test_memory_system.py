from frontend.hetero.memory_system import CanonicalRange, ResidencyManager


def test_explicit_noncoherent_residency_transitions_are_versioned() -> None:
    manager = ResidencyManager()
    manager.register("R0.kv", "shared0.dram3d", "gpu0", 1)
    synchronized = manager.transition(
        CanonicalRange("R0.kv", 0, 0, 64),
        "shared0.dram3d",
        "atlas0.compute",
        "synchronize",
        10,
    )
    assert synchronized.state == "shared_clean"
    written = manager.transition(
        CanonicalRange("R0.kv", 0, 0, 64),
        "shared0.dram3d",
        "atlas0.compute",
        "write",
        20,
    )
    assert written.version == 1
    assert written.owner_device == "atlas0.compute"
    assert len(manager.events) == 2
