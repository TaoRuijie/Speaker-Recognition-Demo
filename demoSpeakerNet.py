import argparse, numpy, soundfile, importlib, warnings, torch, os, glob, pandas

def loadWAV(filename):
	audio, sr = soundfile.read(filename)
	if len(audio.shape) == 2: # dual channel will select the first channel
		audio = audio[:,0]

	feat = numpy.stack([audio],axis=0).astype(numpy.float)
	feat = torch.FloatTensor(feat)
	return feat

def loadPretrain(model, pretrain_model):
	self_state = model.state_dict();
	loaded_state = torch.load(pretrain_model, map_location="cpu")
	for name, param in loaded_state.items():
	    origname = name;    
	    if name not in self_state:
	        name = name.replace("__S__.", "")
	        if name not in self_state:
		        continue;    
	    self_state[name].copy_(param)
	self_state = model.state_dict()
	return model

# Load Model
warnings.filterwarnings("ignore")
SpeakerNetModel = importlib.import_module('models.ResNetSE34V2').__getattribute__('MainModel')
model = SpeakerNetModel()
model = loadPretrain(model, 'models/pretrain.model')

# Load Wav
enroll_audios = glob.glob('data/enroll/*.wav')
test_audios       = glob.glob('data/test/*.wav')

utt_enroll_dict = {}
utt_test_dict = {}
for audio in enroll_audios:
	utt_enroll_dict[audio] = loadWAV(audio)
for audio in test_audios:
	utt_test_dict[audio] = loadWAV(audio)

# Feed data into model, compute the score
model.eval()

with torch.no_grad():
	for audio in test_audios:
		score_matrix = {}
		feat_test   = model(utt_test_dict[audio]).detach()
		feat_test   = torch.nn.functional.normalize(feat_test, p=2, dim=1)
		for enroll_audio in enroll_audios:
			feat_enroll = model(utt_enroll_dict[enroll_audio]).detach()
			feat_enroll = torch.nn.functional.normalize(feat_enroll, p=2, dim=1)
			score = [float(numpy.round(- torch.nn.functional.pairwise_distance(feat_enroll.unsqueeze(-1), feat_test.unsqueeze(-1).transpose(0,2)).detach().numpy(), 4))]
			score_matrix[enroll_audio.split('/')[-1].split('.')[0]] = score
		score_matrix = {k: v for k, v in sorted(score_matrix.items(), key=lambda item: item[1], reverse = True)}
		score_matrix = pandas.DataFrame.from_dict(score_matrix, orient='index', columns=[audio.split('/')[-1].split('.')[0]])
		print("\nSpeaker recognition results for the test audio %s:" %(audio.split('/')[-1].split('.')[0]))
		print(score_matrix)
