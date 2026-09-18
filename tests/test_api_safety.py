"""API safety and output-schema tests. All live transports are mocked."""
from __future__ import annotations
import copy
import contextlib
import io
import json
import os
import socket
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/api'))
import request_io as rio
import run_generation as runner


def completion(m3=False):
    p = {'title':'课后照护服务安排','content':'家长工作时间与孩子放学时间难以衔接，希望有关部门协调提供适当的课后照护服务。'}
    if m3:
        p.update(reflection={'persistent_constraints':[], 'salient_concerns':[], 'current_policy_tensions':[], 'unresolved_uncertainty':[]},
                 planning={'focal_problem':'照护时间衔接','requested_action':'请有关部门协调课后照护服务','supporting_facts':[],'prohibited_claims':[]})
    return {'model':'mock-model','id':'mock-response','choices':[{'finish_reason':'stop','message':{'content':json.dumps(p,ensure_ascii=False),'reasoning_content':'SHOULD_NOT_BE_STORED'}}], 'usage':{'total_tokens':10}}


class APISafety(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.patcher = patch.object(runner, 'RUNS', self.root / 'api_runs')
        self.patcher.start()
        self.env = patch.dict(os.environ, {'DEEPSEEK_API_KEY_SIMULATION':'test-only-credential', 'DEEPSEEK_API_KEY':'test-only-credential','DASHSCOPE_API_KEY':'test-only-credential'})
        self.env.start()
        self.plan = rio.prepare_plan(ROOT / 'examples/api/simulation.jsonl')

    def tearDown(self):
        self.env.stop(); self.patcher.stop(); self.temp.cleanup()

    def paid(self, plan=None, transport=None, **kwargs):
        args = dict(execute=True, confirmation=runner.COST_CONFIRMATION, max_requests=1, run_id='test')
        args.update(kwargs)
        return runner.execute_plan(plan or self.plan, transport=transport or Mock(return_value=completion()), **args)

    def test_default_ignores_configured_credentials_and_transport(self):
        with patch.object(runner, 'credential', side_effect=AssertionError('credential accessed')):
            transport = Mock(side_effect=AssertionError('network attempted'))
            result = runner.execute_plan(self.plan, transport=transport)
        self.assertEqual(result['network_calls'],0); transport.assert_not_called()
        self.assertFalse((self.root / 'api_runs').exists())

    def test_cli_dry_default_with_network_blocked(self):
        with patch.object(socket.socket,'connect',side_effect=AssertionError('network')):
            with patch.object(runner, 'http_once', side_effect=AssertionError('network')):
                for args in [[],['--dry-run'],['--input',str(ROOT/'examples/api/relevance.jsonl')],['--input',str(ROOT/'examples/api/query.jsonl')]]:
                    with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(runner.main(args),0)

    def test_execute_without_confirmation_blocked(self):
        with self.assertRaises(runner.RunStopped):self.paid(confirmation='')
        self.assertFalse((self.root/'api_runs').exists())

    def test_execute_without_cap_blocked(self):
        for cap in [None,0,-1,True]:
            with self.subTest(cap=cap),self.assertRaises(runner.RunStopped):self.paid(max_requests=cap)

    def test_placeholder_key_is_rejected(self):
        with patch.dict(os.environ,{'DEEPSEEK_API_KEY_SIMULATION':'YOUR_API_KEY_HERE'}):
            with self.assertRaises(runner.RunStopped):self.paid()
        self.assertFalse((self.root/'api_runs').exists())

    def test_request_cap_and_resume(self):
        transport=Mock(return_value=completion())
        r=self.paid(transport=transport);self.assertEqual(r['logical_rows_complete'],1)
        self.assertEqual(r['status'],'paused_at_request_cap');self.assertEqual(transport.call_count,1)
        r=self.paid(transport=transport,resume=True);self.assertEqual(r['logical_rows_complete'],2)
        self.assertEqual(transport.call_count,2)

    def test_existing_run_not_overwritten(self):
        self.paid()
        before=(self.root/'api_runs/test/results.jsonl').read_bytes()
        with self.assertRaises(runner.RunStopped):self.paid()
        self.assertEqual(before,(self.root/'api_runs/test/results.jsonl').read_bytes())

    def test_identical_requests_use_cache(self):
        plan=[self.plan[0],copy.deepcopy(self.plan[0])];plan[1]['record_id']='another_logical_row'
        transport=Mock(return_value=completion());r=self.paid(plan,transport)
        self.assertEqual(r['logical_rows_complete'],2);self.assertEqual(transport.call_count,1)
        r=self.paid(plan,transport,resume=True);self.assertEqual(r['wire_attempts_this_invocation'],0)

    def test_changed_inputs_cannot_resume(self):
        self.paid();plan=copy.deepcopy(self.plan);plan[0]['timeout_seconds']=1
        with self.assertRaises(runner.RunStopped):self.paid(plan,resume=True)

    def test_timeout_not_retried_or_logged_with_secrets(self):
        t=Mock(side_effect=TimeoutError('sensitive-request-content test-only-credential'))
        with self.assertRaises(runner.RunStopped):self.paid(transport=t)
        self.assertEqual(t.call_count,1)
        with self.assertRaises(runner.RunStopped):self.paid(transport=t,resume=True)
        self.assertEqual(t.call_count,1)
        for path in (self.root/'api_runs').rglob('*.json*'):
            self.assertNotIn('test-only-credential',path.read_text());self.assertNotIn('sensitive-request-content',path.read_text())

    def test_explicit_retry_failed_is_bounded(self):
        with self.assertRaises(runner.RunStopped):self.paid(transport=Mock(side_effect=TimeoutError()))
        t=Mock(return_value=completion());r=self.paid(transport=t,resume=True,retry_failed=True)
        self.assertEqual(t.call_count,1);self.assertEqual(r['logical_rows_complete'],1)

    def test_inflight_intent_blocks_silent_retry(self):
        with self.assertRaises(KeyboardInterrupt):self.paid(transport=Mock(side_effect=KeyboardInterrupt()))
        t=Mock(return_value=completion())
        with self.assertRaises(runner.RunStopped):self.paid(transport=t,resume=True)
        t.assert_not_called()

    def test_invalid_content_is_not_resampled(self):
        invalid={'choices':[{'finish_reason':'stop','message':{'content':'not json'}}]}
        t=Mock(return_value=invalid)
        with self.assertRaises(runner.RunStopped):self.paid(transport=t)
        with self.assertRaises(runner.RunStopped):self.paid(transport=t,resume=True,retry_failed=True)
        self.assertEqual(t.call_count,1)

    def test_m3_configuration_and_visible_structure(self):
        m3=self.plan[3];b=m3['request']['body']
        self.assertEqual(b['max_tokens'],6144);self.assertEqual(b['reasoning_effort'],'high')
        self.assertNotIn('temperature',b);self.assertIn('reflection',b['messages'][0]['content'])
        with self.assertRaises(ValueError):rio.parse_completion(completion(),m3['request'])
        response=rio.parse_completion(completion(True),m3['request']);self.assertIn('planning',response['parsed'])

    def test_hidden_reasoning_is_not_saved(self):
        self.paid()
        for p in (self.root/'api_runs').rglob('*.json*'):
            self.assertNotIn('SHOULD_NOT_BE_STORED',p.read_text())
            self.assertNotIn('test-only-credential',p.read_text())

    def test_relevance_switches_match_recorded_configuration(self):
        plan=rio.prepare_plan(ROOT/'examples/api/relevance.jsonl')
        self.assertEqual(plan[0]['request']['body']['max_tokens'],800)
        self.assertNotIn('thinking',plan[0]['request']['body'])
        self.assertEqual(plan[1]['request']['body']['thinking'],{'type':'disabled'})
        for row in plan[2:]:self.assertIs(row['request']['body']['enable_thinking'],False)

    def test_query_configuration_and_axes(self):
        plan=rio.prepare_plan(ROOT/'examples/api/query.jsonl');r=plan[0]['request']
        self.assertEqual(r['body']['max_tokens'],1024);self.assertEqual(r['body']['temperature'],0.0)
        self.assertEqual(r['body']['thinking'],{'type':'disabled'})
        with self.assertRaises(ValueError):rio.validate_payload({'queries':[{'concern':'absent','query_text':'问题'}]},r)

    def test_empty_prompt_and_duplicate_id_rejected(self):
        records=rio.read_records(ROOT/'examples/api/simulation.jsonl')
        for value in ['',None,'nan','   ']:
            changed=copy.deepcopy(records);changed[0]['user_prompt']=value
            path=self.root/'bad.jsonl';path.write_text('\n'.join(json.dumps(x) for x in changed))
            with self.assertRaises(ValueError):rio.prepare_plan(path)
        path.write_text('\n'.join(json.dumps(x) for x in [records[0],records[0]]))
        with self.assertRaises(ValueError):rio.prepare_plan(path)

    def test_no_credentials_in_plan_or_url(self):
        with self.assertRaises(ValueError):rio.check_no_credentials({'api_key':'anything'})
        for url in ['http://example.org','https://user:pass@example.org','https://example.org?api_key=x']:
            with self.assertRaises(ValueError):rio.endpoint(url)

    def test_output_name_cannot_escape(self):
        for name in ['../data','a/b','/other','..']:
            with self.assertRaises(runner.RunStopped):self.paid(run_id=name)

    def test_redirect_is_refused(self):
        with self.assertRaises(urllib.error.HTTPError):
            runner.NoRedirect().redirect_request(urllib.request.Request('https://example.invalid'),None,302,'redirect',{},'https://elsewhere.invalid')

    def test_corrupted_cached_results_are_rejected(self):
        self.paid()
        p=next((self.root/'api_runs/test/cache').glob('*.json'))
        if p.name.endswith('.attempt.json'):
            p=next(x for x in (self.root/'api_runs/test/cache').glob('*.json') if not x.name.endswith('.attempt.json'))
        d=json.loads(p.read_text());d['record']['response']['content']='corrupted';p.write_text(json.dumps(d))
        with self.assertRaises(runner.RunStopped):self.paid(resume=True)


if __name__ == '__main__':
    unittest.main()
