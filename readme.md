# Speaker Recognition Demo

This code is modified based on https://github.com/clovaai/voxceleb_trainer , include the pretrain model.

## What it can do ?
This code can output the recognition score between the test utterance (the utterance you want to know the ID) and enrollment utterances (the utterance you have registed the ID) based on the person's unique voiceprint.

## How to use ?
### Input: 
  
  Put the wav files you have registed in the `data/enrollment_audio` folder. Put the wav file you want to test in `data/test_audio` folder (You can put multi test audio file, each test audio will generate one score lists).

### Commend:

```
python demoSpeakerNet.py
```

### Output: 
  
  The score is the speaker recognition score between the test utterance and all enrollment utterances, the score is higher, the utterance tends to come from the same speaker (The highest is 0.0).  
  
  For instance: There are 7 audios in enrollment folder and 1 audio in test folder. You can get the following output:
	
	```
	Speaker recognition results for the test audio A1:
	   |     A1
	A2 | -0.8122
	D2 | -1.2551
	D1 | -1.2641
	B1 | -1.3041
	C2 | -1.3062
	B2 | -1.3246
	C1 | -1.3756
	```

	
`A1` and `A2` come from the same speaker, so they have the highest score.

## Other issue:

- Audio file format

	Current only wav, any sampling rate, single/dual channel is fine. 

- Requirements

	Pytorch, pandas, soundfile package
	
- GPU
	
	This is just the demo code, so the default setting is to use cpu only. It will not be very slow.

- Threshold setting

	It depend on the dataset. Just for suggestion: usually the score between 0 to -1.0 can be viewed as the same speaker, score small than -1.0 can be viewed as different speaker.

- Reference

	Please check their paper for more details:

	```
	@inproceedings{chung2020in,
	  title={In defence of metric learning for speaker recognition},
	  author={Chung, Joon Son and Huh, Jaesung and Mun, Seongkyu and Lee, Minjae and Heo, Hee Soo and Choe, Soyeon and Ham, Chiheon and Jung, Sunghwan and Lee, Bong-Jin and Han, Icksang},
	  booktitle={Interspeech},
	  year={2020}
	}
	```
