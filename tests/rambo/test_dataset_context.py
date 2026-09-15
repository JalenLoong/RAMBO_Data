"""Current-context checks allow explicitly labelled historical and snapshot references."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def test_current_dataset_contract_is_explicit():
    text=(ROOT/'docs/current/adaptation_v2_dataset_contract.md').read_text()
    for value in ('wam-quadruped-v2.1.0','lerobot==0.3.3','limited/tv','CRF 18','N+1','terminal','mask=false','contact_sensor.parquet','标称200Hz','RAMBO_DATA_ROOT','WAM_POLICY_CACHE_ROOT','NEVER_MODEL_INPUT'):
        assert value in text,value
    assert '视频须无损' not in text
    assert 'simulator 仍可使用目标硬件的 optical parameters，同时按照模型需要以 12.5 Hz 采样 / 记录 Dataset RGB' not in text
    assert '先写`*.partial.mp4`' in text
    assert '没有新增workspace bounds' in text


def test_old_rules_are_marked_and_current_entries_route_to_new_contract():
    assert 'DATA-002' in (ROOT/'AGENTS.md').read_text()
    assert 'wam-quadruped-v2.1.0' in (ROOT/'docs/current/contracts-v2.md').read_text()
    assert 'superseded_in_part' in (ROOT/'docs/decisions/ADR-0003.md').read_text()
    for tree,file in [('changes','change-spec.md'),('work','plan.md')]:
        text=(ROOT/f'docs/{tree}/archive/DATA-001/{file}').read_text()
        assert 'Historical DATA-001 acceptance' in text and 'superseded' in text
    assert 'Existing 12.5Hz mounted RGB reference' in (ROOT/'README.md').read_text()
