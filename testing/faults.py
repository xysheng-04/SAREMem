"""Synthetic-only faults. Never imported by the production launcher."""
from dataclasses import dataclass
import json
import time
from types import SimpleNamespace

import numpy as np


class FakeEncoder:
    client = SimpleNamespace(close=lambda: None)
    def encode(self, texts):
        return np.array([[1., 0.] for _ in texts], dtype=np.float32)


class FakeVLM:
    client = SimpleNamespace(close=lambda: None)
    def describe_images(self, *args, **kwargs):
        return []
    def plan_query(self, *args):
        return {'text_anchors':['synthetic bicycle']}
    def rerank(self, text, images, candidates):
        return {c['id']:90-i for i,c in enumerate(candidates)}


class FakeFrozen(FakeEncoder):
    def __init__(self, **kwargs):
        pass
    def _json(self, **kwargs):
        return {'query_anchors':['synthetic bicycle'],'keywords':[], 'visual_description':'',
                'plan':{'schema_version':1,'mode':'direct','capabilities':['direct_fact_lookup'],
                        'required_slots':['fact'],'rerank_policy':'on','tool_calls':[],
                        'assembly_mode':'ranked','seed_reserve':10,'expansion_budget':0,
                        'confidence':.9,'reason_code':'direct_sufficient'}}
    @staticmethod
    def _multimodal_content(text, images):
        return [{'type':'text','text':text}]
    def close(self):
        pass


class FakeAssessment:
    def __init__(self, mode):
        self.mode = mode
    def call(self, stage, payload, schema):
        from aml_api.production.policy import SearchFault
        if self.mode == 'provider':
            raise SearchFault('provider')
        if self.mode == 'schema':
            return {'relevance':'invented'}
        if self.mode == 'id':
            return {'relevance':'direct','fragment_ids':['outside'],'possible_conflict':False}
        if self.mode == 'duplicate_ref':
            return {'relevance':'direct','fragment_ids':['f0','f0'],'possible_conflict':False}
        return {'relevance':'context','fragment_ids':['f0'],'possible_conflict':False}
    def close(self):
        pass


@dataclass
class FaultBackend:
    simple_mode: str = 'success'
    legacy_mode: str = 'success'
    seed_mode: str = 'success'
    raw: dict | None = None

    def __call__(self, stage, settings, request, policy, deadline, checkpoint, publish, stats):
        # No network is possible even if a fixture accidentally reaches a real SDK.
        import socket
        def blocked(*args, **kwargs):
            raise AssertionError('offline test attempted network')
        socket.socket.connect = blocked
        from aml_api.production import runtime as rt
        from aml_api.production.process import execute
        from aml_api.production.policy import SearchFault
        mode = getattr(self, stage+'_mode')
        if stage=='simple' and mode in {'raw','response_error'}:
            from aml_api.production.validation import reader,original
            from aml_api.postseed.seed_port import digest,content_digest
            with reader(settings.storage_path) as db:
                ids=[x['memory_id'] for x in db.execute('SELECT memory_id FROM memories WHERE user_id=? ORDER BY seq',(request.user_id,))]
                publish({'request':digest(request.model_dump(mode='json')),'hits':[{'id':mid,'hash':content_digest(original(db,request.user_id,mid,policy.source_bytes)[0]),'score':1./(i+1)} for i,mid in enumerate(ids)]})

        if mode == 'timeout':
            time.sleep(30)
        if mode in {'error','before_seed'}:
            raise SearchFault('provider')
        if mode == 'source_error':
            raise SearchFault('source_unavailable', terminal=True)
        if mode == 'scope_error':
            raise SearchFault('scope_denied', terminal=True)
        if mode == 'raw':
            return self.raw
        if mode == 'response_error':
            return {'data':[], 'debug':'MUST_NOT_LEAK'}
        if mode == 'empty':
            return {'data':[]}
        if stage == 'simple':
            rt.FrozenProviders = FakeFrozen
            actual_provider = rt.AssessmentProvider
            def assessment(s, controls):
                if mode.startswith('sdk_'):
                    import httpx
                    from openai import OpenAI
                    def respond(req):
                        if mode == 'sdk_timeout':
                            raise httpx.ReadTimeout('MUST_NOT_LEAK',request=req)
                        text = '{invalid' if mode == 'sdk_bad_json' else json.dumps({
                            'relevance':'context','fragment_ids':['f0'],'possible_conflict':False})
                        return httpx.Response(200,json={'id':'synthetic','object':'chat.completion',
                            'created':0,'model':'gpt-4o-mini','choices':[{'index':0,
                            'message':{'role':'assistant','content':text},'finish_reason':'stop'}]})
                    obj = actual_provider.__new__(actual_provider)
                    obj.controls = controls
                    obj.client = OpenAI(api_key='offline-placeholder',base_url='https://invalid.test/v1',
                                        max_retries=0,http_client=httpx.Client(transport=httpx.MockTransport(respond)))
                    return obj
                return FakeAssessment(mode)
            rt.AssessmentProvider = assessment
            original = rt.run_search
            def search(*args):
                if mode=='after_seed_error':raise SearchFault('provider')
                from aml_api.postseed.contracts import Failure
                if mode == 'sort':
                    raise Failure('ranking_contract','sort')
                if mode == 'membership':
                    raise Failure('membership_changed','sort')
                if mode == 'after_seed_timeout':
                    time.sleep(30)
                outcome = original(*args)
                if mode == 'response_validation':
                    outcome.response.data[0].content[0].text = 'MUST_NOT_LEAK'
                return outcome
            rt.run_search = search
        if stage == 'legacy':
            from aml_api.service import RouteAService
            rt.RouteAService = lambda s, store: RouteAService(s, store=store, embedder=FakeEncoder(), vlm=FakeVLM())
        return execute(stage, settings, request, policy, deadline, checkpoint, publish, stats)
