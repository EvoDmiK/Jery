# v4 데이터 보강 비교군

기존 `hermes-ten-tools-2289142ab57a` 카탈로그와 짧은 설명을 유지하고 신규 합성200개를 추가한 비교군이다. `description_control`은 원래 train600, `augmented`는 train800이며 validation/test 각20개의 파일은 v3와 바이트까지 동일하다.

실험 구성·검증·재현 명령은 [v4 README](../hermes-ten-tools-v4/README.md)를 따른다. 이 비교군과 새 설명 비교군은 학습800개를 사용하지만 카탈로그 버전이 다르다. 기존 도구 설명이 담긴 `tools.json`을 새 설명의 모델에 혼용하지 않는다.
