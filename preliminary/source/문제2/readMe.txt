# 문제2. 에러 음성파일 찾기

1. environment
	1) Create conda env
		$ conda create -n q2_env python==3.9
		$ conda activate q2_env

	2) Install required packages using requirements.txt
		$ pip3 install -r requirements.txt

2. how to start
	In the directory where Q2.py exists, enter the following command:
   	$ python3 Q2.py {path-to-wavlist.txt} {path-to-output-file}
	e.g., python3 Q2.py ./wavlist.txt ../../output/Q2.json