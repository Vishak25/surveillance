from surveillance_tf.demo.coord import Configuration, IncidentCoordinator, IncidentState


def test_incident_coordinator_transitions():
    config = Configuration(
        anomaly_threshold=0.75,
        ci_min_for_alert=0.75,
        cooldown_seconds=5,
        window_frames=32,
        stride_frames=3,
        code_red_classes=["Abnormal"],
        code_yellow_classes=[],
    )
    coordinator = IncidentCoordinator(config)
    incident = coordinator.register(timestamp=0.0, mean_conf=0.8, ci_low=0.7, ci_high=0.9, class_name="Abnormal")
    assert incident.severity == "code_red"
    coordinator.advance_state(incident)
    assert incident.state in {IncidentState.READY, IncidentState.ALERTED}
    coordinator.register(timestamp=3.0, mean_conf=0.85, ci_low=0.8, ci_high=0.92, class_name="Abnormal")
    coordinator.advance_state(incident)
    assert incident.state == IncidentState.ALERTED
    coordinator.close_stale(current_ts=12.0)
    assert incident.state == IncidentState.CLOSED
