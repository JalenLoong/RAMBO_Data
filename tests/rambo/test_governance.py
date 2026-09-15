"""Positive/negative tests of governance, using a small isolated documentation tree."""
from pathlib import Path
import copy
import importlib.util
import json
import shutil
import pytest

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'docs/governance/work-registry.json').is_file())
spec=importlib.util.spec_from_file_location('governance_check',ROOT/'scripts/check_governance.py')
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)

@pytest.fixture
def sample(tmp_path):
    for tree in ['governance','templates','changes','work','decisions','current']:
        shutil.copytree(ROOT/'docs'/tree,tmp_path/'docs'/tree)
    for name in ['AGENTS.md','README.md']:shutil.copy2(ROOT/name,tmp_path/name)
    return tmp_path

def test_actual_repository():
    assert not g.check(ROOT)

def test_valid_sample_and_peer(sample,tmp_path):
    peer=tmp_path/'peer';shutil.copytree(sample/'docs/governance',peer/'docs/governance')
    assert not g.check(sample,peer,history=False)

@pytest.mark.parametrize('mutation', ['duplicate','category','gap','alias','namespace'])
def test_bad_registry(mutation):
    registry=json.loads((ROOT/'docs/governance/work-registry.json').read_text())
    if mutation=='duplicate':registry['works'].append(copy.deepcopy(registry['works'][0]))
    if mutation=='category':registry['works'][0]['category']='DATA'
    if mutation=='gap':registry['works'][0]['id']='INFRA-009'
    if mutation=='alias':registry['works'][1]['aliases']=registry['works'][0]['aliases']
    if mutation=='namespace':registry['namespace']='adaptation-v1'
    assert g.registry_errors(registry)

def test_missing_pair(sample):
    next((sample/'docs/work').glob('*/*/plan.md')).unlink()
    assert any('pairing' in e for e in g.check(sample,history=False))

def test_wrong_status(sample):
    path=next((sample/'docs/work').glob('*/*/plan.md'));path.write_text(path.read_text().replace('status: completed','status: abandoned').replace('status: active','status: abandoned'))
    assert any('status' in e for e in g.check(sample,history=False))

def test_peer_conflict(sample,tmp_path):
    peer=tmp_path/'peer';shutil.copytree(sample/'docs/governance',peer/'docs/governance')
    (peer/'docs/governance/work-registry.json').write_text('{}')
    assert any('peer declaration' in e for e in g.check(sample,peer,history=False))

def test_current_identity_and_historical_path():
    assert g.legacy_current_errors('Work ID: V2-DATA-CONTRACT')
    assert not g.legacy_current_errors('Evidence: `runs/audit/v2/V2-DATA-CONTRACT/20260915/`')
    assert not g.legacy_current_errors('Historical alias: V2-CONTRACTS')

def test_source_integrity(sample):
    path=sample/'docs/governance/documentation.md';path.write_text(path.read_text()+'changed')
    assert any('source hash' in e for e in g.check(sample,history=False))

def test_bad_adr(sample):
    path=next((sample/'docs/decisions').glob('*.md'));path.write_text(path.read_text().replace('related_work:', 'wrong_field:'))
    assert any('related_work' in e for e in g.check(sample,history=False))
