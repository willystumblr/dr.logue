# 문제2. 에러 음성파일 찾기

1. environment
	1) Create conda env with environment.yaml
		$ conda env create -f environment.yaml
		$ conda activate q2_env

	2) Just using requirements.txt
		$ pip install -r requirements.txt

2. how to start
	In the directory where Q2.py exists, enter the following command:
   	$ python3 Q2.py {path-to-wavlist.txt} {path-to-output-file}
	e.g., python Q2.py ./wavlist.txt ../../output/Q2.json