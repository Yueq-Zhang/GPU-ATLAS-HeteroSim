from scripts.generate_atlas_qproj_artifact import build_bundle


def test_shape_locked_qproj_bundle_matches_gpu_operator() -> None:
    operator, placement, manifest = build_bundle(
        m_dim=1,
        k_dim=2048,
        n_dim=2048,
        core_count=16,
        tile_m=1,
        tile_k=512,
        tile_n=16,
        element_size=2,
    )

    lowered = manifest["lowering"]
    totals = manifest["expected_totals"]
    assert lowered["per_core_shape"] == {"M": 1, "K": 2048, "N": 128}
    assert lowered["iterations_per_core"] == 32
    assert totals["mac_count"] == 4_194_304
    assert totals["matrix_flop_count"] == 8_388_608
    assert totals["vector_accumulation_op_count"] == 8_192
    assert totals["atlas_reported_operation_count"] == 8_396_800
    assert totals["atlas_dram_request_bytes"] == 8_916_992
    assert len(placement["core_tensor"]) == 16

    task = operator["operator"][0]
    assert task["iteration"] == 32
    input_task, weight_task, output_task = task["execution"]["dram"]
    assert input_task["stride_iter"] == 1
    assert input_task["total_iter"] == 32
    assert weight_task["access_stride_add"] == [1, 4]
    assert output_task["init_iter"] == 5
    assert output_task["stride_iter"] == 4
    assert output_task["total_iter"] == 8
