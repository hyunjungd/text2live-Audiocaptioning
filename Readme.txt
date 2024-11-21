실행 명령어: python train_image.py --example_config golden_horse.yaml --audio_file test3.wav
             python 코드.py 학습에 사용할 설정 파일.yaml 오디오 파일 경로 오디오 파일.wav

위와 같은 실행 명령을 받아 --example_config, --audio_file에 지정된 파일 경로를 따라
지정된 파일을 로드한다. 

오디오 파일에서 오디오 캡션을 생성한 후 golden_horse.yaml의 comp_text와 screen_text로 전달한다. 
전달받은 comp_text와 screen_text 외에도 파일에 직접 입력한 bootstrap_text와 src_text를 읽어와
image_config.yaml에서 학습 설정을 읽어온다. 

이후 train_model(config)을 호출하여 학습을 시작한다.

