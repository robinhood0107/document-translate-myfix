# User Initial Harness And Follow-up Notes

## 목적

- 로컬 전용 `harness_collection/하네스 설계 초기 생각들....txt`에 남아 있던 사용자 메모를 `benchmarking/lab` 브랜치의 문서 자산으로 보관한다.
- 이후 로컬 untracked 메모 파일을 제거해도, 초기 문제 정의와 후속 아이디어가 branch history에 남도록 한다.
- 아이디어 착안자: 사용자

## 메모 성격

이 문서는 benchmark canonical requirement와 별도로 보관하는 사용자 원문 메모다.

- 일부 내용은 이미 완료되었거나 폐기되었다.
- 일부 내용은 후속 시리즈 프로젝트/저장 포맷/알림 기능 아이디어다.
- 따라서 현재 Requirement 1 제품 승격 범위와 직접 일치하지 않는 항목도 함께 포함한다.

## 현재 시점에서의 해석

### 이미 반영되거나 결론이 난 항목

1. `13장 기준 전체 워크플로우 분리` 아이디어
   - benchmark로 검증 완료
2. `OCR stage`와 `Gemma stage`를 분리해 시간 이득이 있는지 검증
   - Requirement 1 성공으로 잠김
3. `MangaLMM + PaddleOCR VL` hybrid 전환 기준
   - benchmark 실패로 종료

### 이후 별도 트랙으로 볼 항목

1. `.seriesctpr` 시리즈 프로젝트 설계
2. `.ctpr` 저장 안정성 개선
3. `ntfy` 연결
4. 텍스트 줄바꿈/글자 크기 마지막 보정
5. 자동번역/배너 중간 취소 예외 처리
6. 추후 클라우드/금융권/redis/gRPC 관련 확장 아이디어

## 사용자 원문 메모 보존본

```text
C:\Users\pjjpj\Desktop\harness_collection\01_requirement_workflow_split_harness.md과 
C:\Users\pjjpj\Desktop\harness_collection\02_requirement_hybrid_ocr_selector_harness.md

13장을 기준으로 생각할꺼야

일단 내 지시사항을 하네스형식의 skill.md처럼 이 작업을 위한 하네스로 정확하게 만든다.

이 과정 전체는 반드시 중간중간마다 문제 해결 명세서로 사용자가 착안해 냈다고 하고 모든 발상부터 측정, 구현방법 효과까지 전부 문서화 시켜서 나중에 포트폴리오로 쓸 수 있을정도로 문서를 각 주제별로 상세하게 적는다.
그리고 이 과정 전체를 효율적으로 다스리기 위해서 전체 파이프라인의 워크 플로우를 전체 프로젝트의 텍스트 감지 -> 전체 프로젝트의 OCR -> 전체 프로젝트의 번역 -> 전체 프로젝트의 인페이팅으로 진행해서,
전체OCR단에서는 OCR 컨테이너만 올리고 번역단에서는 전체 Gemma4만 올리면 속도가 빨라질까? 이러면 vram도 지금보다 더 극한으로 사용해서 성능을 더 확실하게 뽑을 수 있을 것 같긴 한데.
그리고 이렇게 된다면 전체OCR단에는 mangallm이랑 paddleocr_vl가 함께 올라가서 우리가 방금 구현한게 더 효율적으로 시간 아끼면서 동작하게 되지 않음? 
도커의 재기동 문제 때문에 헬스체크로 기다리는 시간이 병목인거 같다는 생각을 계속 하게되서 이게 실제로 효과가 있을지를 알아야 할 것 같아

OCR의 도커기동시간은 앵간하면 1분이면 가동되는데 Gemma4가 기동시간이 적어도 3~4분은 기다려야 되서 이걸 OCR단과 번역단이 서로 상주하지 않게 한다면 더 vram에게 넘기는 방식으로 해서 더 효율적으로 만들어야 하는가 그게 궁금하네
어쨋든 이렇게되면 도커기동시간이 총 몇분이 발생하는지 진입점들을 매우 꼼꼼히 체크포인트 계산해서 총 시간을 계산해야 한다. 시간 측정도 해서 문서에 전부 기록해서 전부 근거를 만들어야 한다.
근거가 중요함

그리고 현재 vram 파악하는 함수를 이용해서 더 효율적으로 vllm을 최대한 병렬처리해서 paddleocr_vl을 이용할 수 있게 되었으니 만약 저 위의 1번 계획이 성공적이라면 mangallm이랑 paddleocr_vl을 동시 상주시켜서도 최대의 성능으로 이용할 수 있을 것 같은데(어쩌피 둘이 동시작업 안함)
13장을 기준으로 mangallm이 판별하낸 것과 실제 텍스트 감지된 상자의 개수의 차이를 싹 비교해서 사용자에게 품질 검수를 맡긴 다음 사용자가 이정도면 ok를 하는 페이지들을 알려주면 이 페이지들을 기준으로 이정도의 일치성이면 mangallm 아니면 paddleocr_vl으로 전환되는 기준을 만들어서 품질과 속도를 동시에 잡을 수도 있을 것 같아.
어쩌피 동시에 상주하게 되니까 작동시간은 거의 차이가 없어지잖아. 지금은 mangallm 단독으로 사용하기에는 p_16.jpg 같은 어려운 페이지는 텍스트 감지와 bbox_2d 생성에 큰 어려움을 겪어서 기존의 텍스트 감지와 paddleocr_vl로 품질을 유지시켜줘야 하기 때문이야.

그리고 기존의 파이프라인과 플로우는 남겨놓아야 해. 따로 구현하는거고 전체 설정창에서 전체 워크플로우를 기존이랑 지금 구현하는 것 둘중에 하나로 고를 수 있게 할꺼야. 이거는 거의 이 프로그램의 근간이기 때문에 설계를 진짜 잘해야 하고 클래스나 객체 설계도 진짜 잘해야 함.
이거 시간 측정해서 페이지가 많아지면 많아질수록 어디에서 시간 병목이 생기고 이걸 어떻게 해결할지 생각해봐라

1. 전체 워크플로우의 변화가 가져다 주는 시간적인 이득이 확실한지, 도커 기동과 타임아웃등의 여러 변수들 그리고 전체OCR단의 도커들이랑 Gemma4가 따로 도커에 올라가고 내려온다면 각자 사용 가능해지는 vram과 ngl의 최대치 증가로 가져오는 속도의 증대가 확실한가
이게 확실하다면 이걸로 구현해서 정확하게 문제 없이 같은 품질이 나오면 성공이다

2. 1번이 제대로 구현이 된다면 mangallm이랑 paddleocr_vl을 동시 상주시켜 13장을 기준으로 mangallm이 판별하낸 것과 실제 텍스트 감지된 상자의 개수의 차이를 싹 비교해서 사용자에게 품질 검수를 맡긴 다음 사용자가 이정도면 ok를 하는 페이지들을 알려주면 이 페이지들을 기준으로 이정도의 일치성이면 mangallm 아니면 paddleocr_vl으로 전환되는 기준을 만들어서 품질과 속도를 동시에 잡는다.



일단 지금시점에서 브랜치 정리하고 머지 한 후에 develop에서 브랜치 새로 판다
시리즈 프로젝트라는 새로운 저장 확장자(.seriesctpr) 및 파일시스템 생성, 대기열 기능을 구현
이것 같은 경우에는 프로젝트(.ctpr)말고 시리즈 프로젝트라는 큰 다른 확장자의 새로운 파일 형식을 만들 예정
이 시리즈 프로젝트의 경우 폴더를 입력받는다.
폴더를 입력받아서 내부에 있는 파일 리스트를 지원하는 형식 전부 불러와서 팝업을 띄운다.
지원하는 파일형식에 대해서 전부 파일을 끝까지 들어가서 찾아낸다. 단 이 경우 파일시스템이 일종의 트리 알고리즘이기 때문에 가장 빠르게 트리를 순회(트레버셜)할 수 있는 알고리즘으로 빠르게 검색해야 한다. 상용 라이브러리에서 가장 빠르고 빅오가 가장 낮은 시간복잡도 낮은 파일 탐색 방법을 사용한다.
이 팝업에서 내가 원하는 드래그로 순서를 변경해서 줄을 맞출 수 있고, 원하는 검색된 파일을 체크해서 시리즈 프로젝트에 넣을 것인지 아닌지 선택할 수 있다.
이후 이 시리즈 프로젝트는 지금 프로젝트의 ui/ux를 거의 전부 계승하지만 사진이 뜨는 것이 아니라 프로젝트가 순서대로 뜨게 되며, 왼쪽 미리보기리스트는 없애고 게시판 형식의 리스트를 ui/ux에 맞게 구성한다.
이것에 대한 디자인은 최대한 지금의 ui/ux와 비슷하되 리스트를 클릭할 경우 해당 세부 프로젝트로 들어오게 되며 이 세부 프로젝트는 기존의 프로젝트파일에 접근하는 것과 완전히 동일하다.
리스트 앞에는 예쁘게 번호가 붙여지며, 번호별로 대기열을 뜻한다. 드래그나 해당 리스트의 번호를 수정하는 것으로 대기열을 변경할 수 있으며
여기서는 전역설정만 설정할 수 있게 해야 해서 우측의 설정들은 전역 설정과 자동 진행 설정들만 들고 온다. 이 부분에 대해서는 하나하나 나에게 질문하고 검사를 받아야 한다. 상세 설정하고 싶으면 세부 프로젝트를 클릭해서 들어가야 한다.
그리고 지금 세부 프로젝트인지 시리즈 프로젝트인지 확실하게 인식할 수 있도록 시인성 있게 제작해야 하며, "앞으로가기("~로 이동" 이건 이전까지 이동한 위치 메모리에서 기억)" "뒤로가기("~로 이동" 이건 이전까지 이동한 위치 메모리에서 기억)" "트리구조로 선택이동(폴더 트리 보여줌)" 이 3개를 좌상단에 어울리게 넣어서 효과적으로 이동할 수 있게 한다
시리즈 프로젝트에서는 프로젝트와 다르게 우상단에 추가적으로 "대기열대로 자동번역" 기능을 넣는다.
이 기능의 경우 지금 설정된 대기열대로 자동적으로 자동 번역을 순차적으로 쭉 실행한다.
그리고 시리즈 프로젝트에서 프로그램에서 리스트의 X 표시나 ui나 드래그로 기존의 다른 프로젝트를 추가하거나 제외할 수 있다. 
이 모든 것은 한 파일로 공유도 할 수 있도록 기존 프로젝트 형식을 확인한 뒤 나에게 질문해라. 이에 맞추어 무결성 있게 관리한다.
이 요구조건에 맞게 하네스 마크다운을 작성하고 나에게 질문을 통해 다시 검수 받은 후 최종 계획까지 작성한다.
궁금한거 있으면 나에게 질문해라

그리고 참고해야 할 것은
다만 “원본 이미지가 안 들어가서 불안정” 쪽은 아니고, **저장 방식과 복원 방식의 민감한 지점**이 몇 군데 있어요.
가장 큰 건 **저장 원자성**입니다. 기존 `.ctpr`가 이미 SQLite일 때는 임시 파일로 완성 후 교체하는 방식이 아니라, 같은 DB에 바로 쓰는 경로를 씁니다. 게다가 SQLite 설정도 `journal_mode=DELETE`, `synchronous=NORMAL`이라서, 저장 도중 앱/OS 강제 종료나 전원 장애가 나면 “마지막 저장분 유실”이나 드물게는 DB 손상 가능성이 있습니다. [project_state_v2.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/app/projects/project_state_v2.py:104) [project_state_v2.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/app/projects/project_state_v2.py:207) [project_state_v2.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/app/projects/project_state_v2.py:412)
두 번째는 **lazy blob + temp dir 의존성**입니다. 프로젝트를 열 때 이미지/패치를 전부 즉시 푸는 게 아니라, temp 경로에 “나중에 필요하면 꺼내 쓰는” 방식으로 등록합니다. 그래서 temp 폴더 정리, 권한 문제, 백신/파일 잠금, 연결 캐시가 꼬이는 상황에서는 특정 페이지나 patch가 뒤늦게 안 풀리는 형태로 흔들릴 수 있습니다. [project_state_v2.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/app/projects/project_state_v2.py:146) [project_state_v2.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/app/projects/project_state_v2.py:161) [project_state_v2.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/app/projects/project_state_v2.py:463) [image.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/app/controllers/image.py:340)
세 번째는 **파일이 커질수록 저장 스트레스가 급격히 커진다**는 점입니다. 이 포맷은 원본 이미지만 저장하는 게 아니라, 히스토리 이미지와 patch PNG까지 같이 넣습니다. 그래서 오래 작업한 프로젝트일수록 `.ctpr`가 커지고 저장 시간이 늘어나며, 그만큼 중간 실패 창도 커집니다. [project_state_v2.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/app/projects/project_state_v2.py:304) [project_state_v2.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/app/projects/project_state_v2.py:313) [project_state_v2.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/app/projects/project_state_v2.py:323)
네 번째는 **undo/redo 히스토리는 저장되지 않는다**는 점입니다. 프로젝트를 다시 열면 화면 상태는 복원되지만, undo stack 객체는 새로 만듭니다. 그래서 “열자마자 이전 세션의 작업을 한 단계씩 되돌린다”는 건 안 됩니다. 데이터 손상은 아니지만 사용자는 불안정하게 느낄 수 있어요. [projects.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/app/controllers/projects.py:1104) [image.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/app/controllers/image.py:1327) [projects.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/app/controllers/projects.py:919)
다섯 번째는 **페이지 식별이 path 기반**이라는 점입니다. `page_state`의 기본 키가 UUID가 아니라 `page_path` 문자열입니다. 일반 사용에선 잘 돌아가지만, 프로젝트를 외부에서 가공하거나 특이한 이식/복구를 할 때는 고정 ID 방식보다 덜 단단합니다. [project_state_v2.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/app/projects/project_state_v2.py:65) [project_state_v2.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/app/projects/project_state_v2.py:348)
여섯 번째는 **자동저장이 스냅샷형**이라는 점입니다. 실시간 autosave도 debounce가 있고, recovery 파일도 별도 시점 저장이라서 크래시 직전의 마지막 몇백 ms~몇 초 편집은 빠질 수 있습니다. [projects.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/app/controllers/projects.py:295) [projects.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/app/controllers/projects.py:372) [projects.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/app/controllers/projects.py:430)
정리하면, **가장 실제 위험한 건 “기존 SQLite `.ctpr`에 대한 in-place save”**이고, 그 다음이 **lazy materialization 구조**, 그 다음이 **대형 프로젝트의 blob/history 누적**입니다. 반대로 “원본 이미지가 안 저장돼서 약하다”는 건 아니고, 그 부분은 오히려 꽤 안전한 편입니다.
원하면 다음엔 제가 이걸 기준으로
1. 실제 위험도 순위
2. 바로 손볼 만한 안정화 포인트
3. 호환성 안 깨고 고치는 순서
까지 바로 계획으로 뽑아드릴게요.
이걸 좀 고려해보기
설계를 어떻게 할 것인지, 너무 복잡하게 하지 말고 그냥 파일만 공유하면 그대로 이어서 할 수 있을 정도로만 .ctpr, .seriesctpr를 무결성과 강건성을 키우고 싶거든
궁금한거 있으면 나에게 질문해라
끝나고 브랜치 정리하고 머지 한다.



ntfy의 연결

텍스트 조금 만 더 말풍선으로 들어오게 하려면? 구조적 변화X 무조건 제일 마지막에 그냥 줄바꿈만 어떻게 할 것인지. 그리고 글자크기는 어떻게 할 것인지 현재 알고리즘 확인 후 어떻게 할 것인지 결정

한페이지랑 자동번역 배너로 중간 취소시 뭐 이상한 오류 뜨는데 그거 예외처리로 묶던지 아님 함수단에서 해결을 하던가 



추후 작업)
이제 내 프로젝트로 둔갑시키는 작업이 필요함...
클라우드랑 금융권 관련해서 필요한 지식이나 아키텍처로 redis나 gprc 같은거로 어떻게 할지 생각좀 하기
```
