import argparse, numpy, soundfile, importlib, warnings, torch, os

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

# Definition
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description = "DemoSpeakerRecognition");
parser.add_argument('--enrollment_audio',   type=str,   default=None,   help='Enrollment audio path')
parser.add_argument('--test_audio',         type=str,   default=None,   help='Test audio path')
args = parser.parse_args()

# Load Model
SpeakerNetModel = importlib.import_module('models.ResNetSE34V2').__getattribute__('MainModel')
model = SpeakerNetModel()
model = loadPretrain(model, 'models/pretrain.model')

# Load Wav
utt_enroll = loadWAV(args.enrollment_audio)
utt_test   = loadWAV(args.test_audio)

# Feed data into model, compute the score
model.eval()
with torch.no_grad():
	feat_enroll = model(utt_enroll).detach()
	feat_test   = model(utt_test).detach()
	feat_enroll = torch.nn.functional.normalize(feat_enroll, p=2, dim=1)
	feat_test   = torch.nn.functional.normalize(feat_test, p=2, dim=1)
	score = - torch.nn.functional.pairwise_distance(feat_enroll.unsqueeze(-1), feat_test.unsqueeze(-1).transpose(0,2)).detach().numpy()
	
print("Enroll_audio:                  %s\n\
Test_audio:                    %s\n\
Score (Higher is better):      %.4f"%(args.enrollment_audio, args.test_audio, score))