"""Three-GPU resource layout for the 9B whitebox online pilot."""

from verl.trainer.ppo.utils import Role


def separate_reference_layout(classes, mapping, *, actor_gpus, nodes):
    """Split the fused actor/rollout/reference worker into two Ray pools."""
    if actor_gpus != 2 or nodes != 1:
        raise ValueError("three-GPU pilot requires one node and two actor GPUs")
    worker = classes.pop(Role.ActorRolloutRef)
    mapping.pop(Role.ActorRolloutRef)
    classes[Role.ActorRollout] = worker
    classes[Role.RefPolicy] = worker
    mapping[Role.ActorRollout] = "global_pool"
    mapping[Role.RefPolicy] = "reference_pool"
    return {"global_pool": [2], "reference_pool": [1]}


def validate_device_assignments(actor_ids, reference_ids, visible_ids):
    """Reject accidental colocations or a mismatch with the authorized mask."""
    actor, reference, visible = map(lambda values: set(map(str, values)),
                                    (actor_ids, reference_ids, visible_ids))
    if len(actor_ids) != 2 or len(actor) != 2 or len(reference_ids) != 1 or len(reference) != 1:
        raise ValueError("expected two distinct actor devices and one reference device")
    if actor & reference:
        raise ValueError("actor and reference devices must be disjoint")
    if len(visible_ids) != 3 or actor | reference != visible:
        raise ValueError("allocated devices do not match the three visible devices")
