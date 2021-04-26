# Speaker Verification Demo

This code is modified based on https://github.com/clovaai/voxceleb_trainer The model used here is also the pretrain model in that repositories.

## Introduction:
Speaker verification is to judge if two utterance belong to the same speaker based on the each person's unique voiceprint 

## How to use:
Input: Put the wav files into the data folder, change the name in run.sh file. One is for enrollment_audio, another is for test_audio.

```
bash run.sh
```

Output: The score is the speaker verification score between two utterance, the score is higher, the utterance tends to come from the same speaker (The highest is 0.0)


## Other issue:

1. Is there any requirement for the wav file ?
	No, any sampling rate, single/dual channel is fine. 

2. Requirement
	Pytorch, soundfile package

3. Threshold setting
	It depend on the different dataset. Just for suggestion: usually the score between 0 to -1.0 can be viewed as the same speaker, score small than -1.0 can be viewed as different speaker.

4. Reference
	Please check their paper for more details:
	[1] _In defence of metric learning for speaker recognition_
	```
	@inproceedings{chung2020in,
	  title={In defence of metric learning for speaker recognition},
	  author={Chung, Joon Son and Huh, Jaesung and Mun, Seongkyu and Lee, Minjae and Heo, Hee Soo and Choe, Soyeon and Ham, Chiheon and Jung, Sunghwan and Lee, Bong-Jin and Han, Icksang},
	  booktitle={Interspeech},
	  year={2020}
	}
	```

	[2] _Clova baseline system for the VoxCeleb Speaker Recognition Challenge 2020_
	```
	@article{heo2020clova,
	  title={Clova baseline system for the {VoxCeleb} Speaker Recognition Challenge 2020},
	  author={Heo, Hee Soo and Lee, Bong-Jin and Huh, Jaesung and Chung, Joon Son},
	  journal={arXiv preprint arXiv:2009.14153},
	  year={2020}
	}
	```