# Speaker Verification Demo

This code is modified based on https://github.com/clovaai/voxceleb_trainer , include the pretrain model.

Speaker verification&recognition is to judge if two utterance belong to the same speaker based on the each person's unique voiceprint 

## How to use:
### Input: 
  
  Put the wav files you want to verify into the `data` folder, change the `enrollment_audio` and `test_audio` in `run.sh` file.

### Commend:

```
bash run.sh
```

### Output: 
  
  The score is the speaker verification score between two utterances, the score is higher, the utterance tends to come from the same speaker (The highest is 0.0).  
  
  For instance: Score for `A1.wav` and `A2.wav` is -0.8122 (Same speaker).  `A1.wav` and `B1.wav` is -1.3041 (Different speaker).

## Other issue:

- Audio file format

	Current only wav, any sampling rate, single/dual channel is fine. 

- Requirements

	Pytorch, soundfile package

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
