def test_dot_access(cfg):
    assert cfg.model.variant == "head"
    assert cfg.fit.stages.rigid.iters > 0
    assert cfg.mouth.mode == "pin"


def test_get_with_default(cfg):
    assert cfg.get("fit.huber_delta") == cfg.fit.huber_delta
    assert cfg.get("does.not.exist", 7) == 7


def test_stage_weights_present(cfg):
    for stage in (cfg.fit.stages.rigid, cfg.fit.stages.shape, cfg.fit.stages.expression):
        w = stage.weights.to_dict()
        assert set(w) >= {"lmk", "pho", "psi", "beta", "temp", "mouth", "col"}
