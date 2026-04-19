import numpy as np
import json
import cv2
import requests
import base64
import time
import uuid
import os
from typing import List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base import register_OCR, OCRBase, TextBlock
from utils.message import create_info_dialog


@register_OCR('paddle_vl')
class OCRPaddleVL(OCRBase):
    SERVER_URL = 'http://127.0.0.1:28118/layout-parsing'
    PERF_DEFAULTS = {
        'parallel_workers': 8,
        'max_new_tokens': 1024,
    }
    LIMITS = {
        'parallel_workers': (1, 8),
        'max_new_tokens': (128, 2048),
    }

    params = {
        'server_url': {
            'type': 'text',
            'size': 'long',
            'value': SERVER_URL,
            'description': 'Paddle-VL /layout-parsing 엔드포인트 주소',
        },
        'prettifyMarkdown': {'type': 'checkbox', 'value': False},
        'visualize': {'type': 'checkbox', 'value': False},
        'max_new_tokens': {
            'type': 'line_editor',
            'value': PERF_DEFAULTS['max_new_tokens'],
            'description': '최대 생성 토큰 수 (권장 1024, 범위 128~2048)'
        },
        'parallel_workers': {
            'type': 'line_editor',
            'value': PERF_DEFAULTS['parallel_workers'],
            'description': '블록 OCR 동시 요청 수 (권장 4~8, 단일 사용자 최대 처리량)'
        },
        'trace_logging': {
            'type': 'checkbox',
            'value': True,
            'description': 'Paddle-VL 처리 추적 로그 출력 (요청/응답/파싱 경로)',
        },
        'trace_preview_chars': {
            'type': 'line_editor',
            'value': 160,
            'description': '추적 로그 텍스트 미리보기 길이',
        },
        'description': '로컬 배포 Paddle OCR-VL 서비스 (POST /layout-parsing)'
    }

    @property
    def server_url(self):
        val = self.params.get('server_url')
        # UI may wrap param as a dict like {'value': 'http://...', 'data_type': <class 'str'>}
        if isinstance(val, dict):
            url = val.get('value') or val.get('text') or ''
        else:
            url = val or ''
        return str(url).strip() or self.SERVER_URL

    @property
    def prettifyMarkdown(self):
        v = self.params.get('prettifyMarkdown')
        if isinstance(v, dict):
            return bool(v.get('value', False))
        return bool(v)

    @property
    def parallel_workers(self) -> int:
        d = self.PERF_DEFAULTS['parallel_workers']
        min_v, max_v = self.LIMITS['parallel_workers']
        v = self.params.get('parallel_workers', d)
        if isinstance(v, dict):
            v = v.get('value', d)
        try:
            v = int(v)
        except Exception:
            v = d
        return max(min_v, min(v, max_v))

    @property
    def max_new_tokens(self) -> int:
        d = self.PERF_DEFAULTS['max_new_tokens']
        min_v, max_v = self.LIMITS['max_new_tokens']
        v = self.params.get('max_new_tokens', d)
        if isinstance(v, dict):
            v = v.get('value', d)
        try:
            v = int(v)
        except Exception:
            v = d
        return max(min_v, min(v, max_v))

    @property
    def visualize(self):
        v = self.params.get('visualize')
        if isinstance(v, dict):
            return bool(v.get('value', False))
        return bool(v)

    @property
    def trace_logging(self) -> bool:
        env = os.getenv('BALLOONTRANS_PADDLE_VL_TRACE')
        if env is not None:
            return str(env).strip().lower() not in {'0', 'false', 'off', 'no'}
        v = self.params.get('trace_logging')
        if isinstance(v, dict):
            return bool(v.get('value', True))
        if v is None:
            return True
        return bool(v)

    @property
    def trace_preview_chars(self) -> int:
        d = 160
        v = self.params.get('trace_preview_chars', d)
        if isinstance(v, dict):
            v = v.get('value', d)
        try:
            v = int(v)
        except Exception:
            v = d
        return max(40, min(v, 2000))

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.debug = False

    def _trace(self, msg: str):
        if self.trace_logging:
            self.logger.info(f'[Paddle-VL] {msg}')

    def _preview(self, text: Any) -> str:
        if text is None:
            return ''
        s = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
        s = s.replace('\r', '\\r').replace('\n', '\\n')
        n = self.trace_preview_chars
        if len(s) > n:
            return s[:n] + '...'
        return s

    def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs):
        """
        각 텍스트 블록을 개별로 잘라 로컬 Paddle-VL 서비스에 요청합니다.
        기존 블록 기반 워크플로(TextBlock API)와의 호환성을 유지합니다.
        """
        im_h, im_w = img.shape[:2]
        jobs = []
        for idx, blk in enumerate(blk_list):
            x1, y1, x2, y2 = blk.xyxy
            x1 = max(0, min(int(round(float(x1))), im_w - 1))
            y1 = max(0, min(int(round(float(y1))), im_h - 1))
            x2 = max(x1 + 1, min(int(round(float(x2))), im_w))
            y2 = max(y1 + 1, min(int(round(float(y2))), im_h))
            if x1 < x2 and y1 < y2:
                crop = img[y1:y2, x1:x2]
                jobs.append((idx, blk, crop, (x1, y1, x2, y2)))
            else:
                self.logger.warning('invalid textbbox to target img')
                blk.text = ['']

        if not jobs:
            self._trace('_ocr_blk_list: no valid crops')
            return

        worker_count = min(self.parallel_workers, len(jobs))
        self._trace(f'_ocr_blk_list: jobs={len(jobs)} workers={worker_count}')
        if worker_count <= 1:
            for job_idx, blk, crop, bbox in jobs:
                try:
                    result = self.ocr(crop, trace_ctx=f'blk#{job_idx} bbox={bbox} shape={crop.shape}')
                    blk.text = [result] if result else ['']
                except Exception:
                    self.logger.exception(f'Paddle-VL 블록 단위 인식 실패 (blk#{job_idx}, bbox={bbox})')
                    blk.text = ['']
            return

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(self.ocr, crop, f'blk#{job_idx} bbox={bbox} shape={crop.shape}'): (job_idx, blk, bbox)
                for job_idx, blk, crop, bbox in jobs
            }
            for future in as_completed(future_map):
                job_idx, blk, bbox = future_map[future]
                try:
                    result = future.result()
                    blk.text = [result] if result else ['']
                except Exception:
                    self.logger.exception(f'Paddle-VL 블록 단위 인식 실패 (blk#{job_idx}, bbox={bbox})')
                    blk.text = ['']
        self._trace(f'_ocr_blk_list: completed jobs={len(jobs)}')

    def ocr_img(self, img: np.ndarray) -> str:
        self.logger.debug(f'ocr_img: {img.shape}')
        return self.ocr(img, trace_ctx=f'full_image shape={img.shape}')

    def _extract_texts_from_pruned(self, pruned: Any) -> List[str]:
        texts: List[str] = []

        def walk(node: Any):
            if node is None:
                return
            if isinstance(node, dict):
                # common keys may include 'texts' or 'text'
                if 'texts' in node and isinstance(node['texts'], (list, str)):
                    if isinstance(node['texts'], list):
                        texts.append(''.join(node['texts']).strip())
                    else:
                        texts.append(str(node['texts']).strip())
                if 'text' in node and isinstance(node['text'], str):
                    texts.append(node['text'].strip())
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for it in node:
                    walk(it)
            elif isinstance(node, str):
                texts.append(node.strip())

        walk(pruned)
        # filter empties and deduplicate nearby
        return [t for t in texts if t]

    def _markdown_to_text(self, md: str) -> str:
        """
        Markdown을 간단히 순수 텍스트로 변환합니다:
        - 이미지 문법 ![...](...) 제거
        - 링크 [text](url) -> text 로 변환
        - 제목 앞의 # 제거
        - 강조 기호(*, _, **) 제거
        - 인라인 코드와 HTML 태그 제거
        - 연속 빈 줄 병합 및 앞뒤 공백 제거
        """
        if not md:
            return ''
        try:
            import re

            # remove image markdown
            md = re.sub(r'!\[[^\]]*\]\([^\)]*\)', '', md)
            # replace links [text](url) -> text
            md = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', md)
            # remove heading markers at line starts
            md = re.sub(r'(?m)^\s{0,3}#{1,6}\s*', '', md)
            # remove bold/italic markers (*, _, **, __)
            md = re.sub(r'(\*\*|__)(.*?)\1', r'\2', md)
            md = re.sub(r'(\*|_)(.*?)\1', r'\2', md)
            # remove inline code backticks
            md = re.sub(r'`([^`]*)`', r'\1', md)
            # remove any remaining html tags
            md = re.sub(r'<[^>]+>', '', md)
            # normalize whitespace and remove multiple blank lines
            md = re.sub(r"\r\n|\r", "\n", md)
            md = re.sub(r"\n{2,}", "\n", md)
            md = md.strip()
            return md
        except Exception:
            return md

    def ocr(self, img: np.ndarray, trace_ctx: str = '') -> str:
        """
        이미지를(전체 또는 블록) Base64로 `/layout-parsing` 엔드포인트에 전송합니다.
        응답의 Markdown 텍스트를 우선 사용하고, 없으면 prunedResult에서 텍스트를 추출합니다.
        반환값은 문자열(해당 블록의 인식 결과)입니다.
        """
        req_id = uuid.uuid4().hex[:8]
        t0 = time.perf_counter()
        try:
            image_bytes = cv2.imencode('.jpg', img)[1].tobytes()
        except Exception:
            self.logger.exception(f'이미지 인코딩 실패 (req={req_id}, ctx={trace_ctx})')
            raise

        image_b64 = base64.b64encode(image_bytes).decode('ascii')

        payload = {
            'file': image_b64,
            'fileType': 1,
            'prettifyMarkdown': self.prettifyMarkdown,
            'visualize': self.visualize,
            'maxNewTokens': self.max_new_tokens,
        }
        self._trace(
            f"req={req_id} start ctx='{trace_ctx}' "
            f"shape={getattr(img, 'shape', None)} bytes={len(image_bytes)} "
            f"maxNewTokens={payload.get('maxNewTokens')} prettify={payload.get('prettifyMarkdown')} visualize={payload.get('visualize')}"
        )

        try:
            resp = requests.post(self.server_url, json=payload, timeout=60)
            self._trace(f"req={req_id} http_status={resp.status_code} elapsed_ms={(time.perf_counter() - t0) * 1000:.1f}")
        except Exception:
            self.logger.exception(f'로컬 Paddle-VL 서비스 요청 실패 (req={req_id}, ctx={trace_ctx})')
            raise

        # Some server builds may reject maxNewTokens; retry once with legacy payload.
        if resp.status_code != 200:
            try:
                self._trace(f'req={req_id} retry_without_maxNewTokens due_to_status={resp.status_code}')
                legacy_payload = dict(payload)
                legacy_payload.pop('maxNewTokens', None)
                resp = requests.post(self.server_url, json=legacy_payload, timeout=60)
                self._trace(
                    f"req={req_id} retry_status={resp.status_code} "
                    f"elapsed_ms={(time.perf_counter() - t0) * 1000:.1f}"
                )
            except Exception:
                pass

        if resp.status_code != 200:
            self.logger.error(f'Paddle-VL 요청 실패, 상태 코드: {resp.status_code}')
            try:
                self._trace(f"req={req_id} non200_body='{self._preview(resp.text)}'")
            except Exception:
                pass
            raise ValueError(f'Paddle-VL 요청 실패, 상태 코드: {resp.status_code}')

        try:
            data = resp.json()
        except Exception:
            self.logger.exception(f'Paddle-VL 응답 JSON 파싱 실패 (req={req_id})')
            try:
                self._trace(f"req={req_id} invalid_json_body='{self._preview(resp.text)}'")
            except Exception:
                pass
            raise
        self._trace(f"req={req_id} parsed_json keys={list(data.keys()) if isinstance(data, dict) else type(data)}")

        # Paddle 표준 응답 형식: { logId, errorCode, errorMsg, result }
        if 'errorCode' in data and data.get('errorCode', -1) != 0:
            msg = data.get('errorMsg', '')
            self._trace(f"req={req_id} errorCode={data.get('errorCode')} errorMsg='{self._preview(msg)}'")
            # Some builds report unknown-field style errors for maxNewTokens.
            if 'maxNewTokens' in str(msg):
                legacy_payload = dict(payload)
                legacy_payload.pop('maxNewTokens', None)
                self._trace(f"req={req_id} retry_without_maxNewTokens due_to_errorCode")
                resp_legacy = requests.post(self.server_url, json=legacy_payload, timeout=60)
                if resp_legacy.status_code == 200:
                    data = resp_legacy.json()
                    self._trace(
                        f"req={req_id} legacy_retry_json keys={list(data.keys()) if isinstance(data, dict) else type(data)}"
                    )
                    if 'errorCode' in data and data.get('errorCode', -1) != 0:
                        legacy_msg = data.get('errorMsg', '')
                        self.logger.error(f'Paddle-VL 반환 오류: {legacy_msg}')
                        raise ValueError(f'Paddle-VL 반환 오류: {legacy_msg}')
                else:
                    self.logger.error(f'Paddle-VL 요청 실패, 상태 코드: {resp_legacy.status_code}')
                    raise ValueError(f'Paddle-VL 요청 실패, 상태 코드: {resp_legacy.status_code}')
            else:
                self.logger.error(f'Paddle-VL 반환 오류: {msg}')
                raise ValueError(f'Paddle-VL 반환 오류: {msg}')

        result = data.get('result', data)
        lprs = result.get('layoutParsingResults') or []
        self._trace(
            f"req={req_id} layoutParsingResults={len(lprs)} "
            f"result_keys={list(result.keys()) if isinstance(result, dict) else type(result)}"
        )
        if not lprs:
            # layoutParsingResults가 없으면 result에서 직접 파싱을 시도합니다.
            # 마지막에는 디버깅을 위해 전체 응답 문자열을 반환합니다.
            self.logger.debug('layoutParsingResults를 찾지 못해 전체 응답 문자열을 반환합니다')
            self._trace(
                f"req={req_id} parse_path=fallback_result_json "
                f"elapsed_ms={(time.perf_counter() - t0) * 1000:.1f}"
            )
            return json.dumps(result, ensure_ascii=False)

        first = lprs[0]
        # Markdown 결과를 우선 사용하되 순수 텍스트로 정리합니다.
        md_raw = first.get('markdown', {}).get('text') if isinstance(first.get('markdown'), dict) else None
        if md_raw:
            md_txt = self._markdown_to_text(md_raw)
            if md_txt:
                self._trace(
                    f"req={req_id} parse_path=markdown text_len={len(md_txt)} "
                    f"text_preview='{self._preview(md_txt)}' elapsed_ms={(time.perf_counter() - t0) * 1000:.1f}"
                )
                return md_txt

        # 없으면 prunedResult에서 texts 필드 추출을 시도합니다.
        pruned = first.get('prunedResult')
        if pruned is not None:
            texts = self._extract_texts_from_pruned(pruned)
            if texts:
                # join and clean result to remove any possible markdown artifacts
                joined = '\n'.join(texts)
                parsed = self._markdown_to_text(joined)
                self._trace(
                    f"req={req_id} parse_path=prunedResult texts={len(texts)} text_len={len(parsed)} "
                    f"text_preview='{self._preview(parsed)}' elapsed_ms={(time.perf_counter() - t0) * 1000:.1f}"
                )
                return parsed

        # 마지막 폴백: outputImages/pruned를 포함한 JSON 문자열을 반환합니다.
        self._trace(
            f"req={req_id} parse_path=fallback_first_json "
            f"first_keys={list(first.keys()) if isinstance(first, dict) else type(first)} "
            f"elapsed_ms={(time.perf_counter() - t0) * 1000:.1f}"
        )
        return json.dumps(first, ensure_ascii=False)

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        # server_url 등 파라미터 변경 시 사용자에게 알립니다.
        if param_key == 'server_url':
            create_info_dialog('Paddle-VL 서비스 주소가 업데이트되었습니다')
