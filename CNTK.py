import cupy as cp
import numpy as np
import argparse
import scipy.linalg
from utils import load_cifar
from PIL import Image
from time import time

parser = argparse.ArgumentParser(description = 'Convolutional Neural Tangent Kernel (CNTK) for CIFAR-10')
parser.add_argument('--depth', default = 21, type = int, help = 'depth of CNTK (#conv layers + 1)')
parser.add_argument('--num_train', default = 100, type = int, help = 'number of training samples')
parser.add_argument('--gap', default = "yes", type = str, help = 'whether GAP (global average pooling) is used')
parser.add_argument('--fix', default = "yes", type = str, help = 'whether first layer and last layer are fixed (or trained) (see Section 4.2 in our paper)')
args = parser.parse_args()

d = args.depth
gap = (args.gap == "yes")
fix = (args.fix == "yes")

#CUDA kernel for convolution operation
conv3 = cp.RawKernel(r'''
extern "C" __global__
void conv3(const float s[32][32][32][32], float t[32][32][32][32])
{
	int x1 = threadIdx.x + blockIdx.x - 31;
	int y1 = threadIdx.y + blockIdx.y - 31;
	int x2 = threadIdx.x;
	int y2 = threadIdx.y;

	__shared__ float d[32 + 2][32 + 2];
	if (x2 == 0){
		d[0][y2 + 1] = d[33][y2 + 1] = 0;
		if (x2 == 0 && y2 == 0)
			d[0][0] = d[0][33] = d[33][0] = d[33][33] = 0; 
	}
	if (y2 == 0){
		d[x2 + 1][0] = d[x2 + 1][33] = 0;
	}

	if (x1 < 0 || x1 > 31 || y1 < 0 || y1 > 31){
		d[x2 + 1][y2 + 1] = 0;
		return;
	}
	else
		d[x2 + 1][y2 + 1] = s[x1][y1][x2][y2];
	__syncthreads();

	t[x1][y1][x2][y2] = d[x2][y2] + d[x2][y2 + 1] + d[x2][y2 + 2]
					  + d[x2 + 1][y2] + d[x2 + 1][y2 + 1] + d[x2 + 1][y2 + 2]
					  + d[x2 + 2][y2] + d[x2 + 2][y2 + 1] + d[x2 + 2][y2 + 2];

}''', 'conv3')
conv_blocks = (63, 63)
conv_threads = (32, 32)

#CUDA kernel for activation
trans = cp.RawKernel(r'''
extern "C" __global__
void trans(float s[32][32][32][32], float t[32][32][32][32], const float l[32][32], const float r[32][32], const float il[32][32], const float ir[32][32])
{
	int x1 = blockIdx.x;
	int y1 = blockIdx.y;
	int x2 = threadIdx.x + ((blockIdx.z >> 2) << 3);
	int y2 = threadIdx.y + ((blockIdx.z & 3) << 3);
	float S = s[x1][y1][x2][y2], T = t[x1][y1][x2][y2], L = l[x1][y1], R = r[x2][y2], iL = il[x1][y1], iR = ir[x2][y2];
	S = S * iL * iR;
	float BS = (S * (3.141592654f - acosf(max(min(S, 1.0f), -1.0f))) + sqrtf(1.0f - min(S * S, 1.0f))) * L * R / 28.274333882308138f;
	S = (3.141592654f - acosf(max(min(S, 1.0f), -1.0f))) / 28.274333882308138;
	t[x1][y1][x2][y2] = T * S + BS;
	s[x1][y1][x2][y2] = BS;

}''', 'trans')
trans_blocks = (32, 32, 16)
trans_threads = (8, 8)

#Calculate diagonal entries of $\Sigma^{(h)}(x, x)$ and their reciprocals. See Section 4.3 in our paper. 
def xx(x):
	RL = [1.0, ]
	iRL = [1.0, ]
	S = cp.matmul(x.T, x).reshape(32, 32, 32, 32)
	conv3(conv_blocks, conv_threads, (S, S))
	T = cp.zeros((32, 32, 32, 32), dtype = cp.float32)
	if not fix:
		T += S
	for i in range(1, d - 1):
		L = cp.sqrt(cp.diag(S.reshape(1024, 1024)).reshape(32, 32))
		iL = 1.0 / L
		RL.append(L)
		iRL.append(iL)
		trans(trans_blocks, trans_threads, (S, T, L, L, iL, iL))		
		conv3(conv_blocks, conv_threads, (S, S))
		conv3(conv_blocks, conv_threads, (T, T))
	L = cp.sqrt(cp.diag(S.reshape(1024, 1024)).reshape(32, 32))
	iL = 1.0 / L
	RL.append(L)
	iRL.append(iL)
	trans(trans_blocks, trans_threads, (S, T, L, L, iL, iL))	
	if fix:
		T -= S
	return RL, iRL

#Caclulate the kernel value of x and z.
#Lx and Lz are diagonal entries of $\Sigma^{(h)}(x, x)$ and $\Sigma^{(h)}(z, z)$. 
#iLx and iLz are reciprocals of diagonal entries of $\Sigma^{(h)}(x, x)$ and $\Sigma^{(h)}(z, z)$. 
def xz(x, z, Lx, Lz, iLx, iLz):
	S = cp.matmul(x.T, z).reshape(32, 32, 32, 32)
	conv3(conv_blocks, conv_threads, (S, S))
	T = cp.zeros((32, 32, 32, 32), dtype = cp.float32)
	if not fix:
		T += S
	for i in range(1, d - 1):
		trans(trans_blocks, trans_threads, (S, T, Lx[i], Lz[i], iLx[i], iLz[i]))		
		conv3(conv_blocks, conv_threads, (S, S))
		conv3(conv_blocks, conv_threads, (T, T))
	trans(trans_blocks, trans_threads, (S, T, Lx[-1], Lz[-1], iLx[-1], iLz[-1]))	
	if fix:
		T -= S	
	return cp.mean(T) if gap else cp.trace(T.reshape(1024, 1024))

#Load CIFAR-10.
(X_train_all, y_train_all), (X_test, y_test) = load_cifar('/fs/vulcan-datasets/cifar-10-python')

for trial_i in range(5):
	begin_time = time()
	X_train = X_train_all[trial_i*args.num_train:(trial_i+1)*args.num_train]
	y_train = y_train_all[trial_i*args.num_train:(trial_i+1)*args.num_train]
	# X_test = X_test[:100]
	# y_test = y_test[:100]
	X = np.concatenate((X_train, X_test), axis = 0)
	N = X.shape[0]
	N_train = X_train.shape[0]
	N_test = X_test.shape[0]
	X = cp.asarray(X).reshape(-1, 3, 1024)
	#Calculate diagonal entries.
	L = []
	iL = []
	for i in range(N):
		# print(i)
		Lx, iLx = xx(X[i])	
		L.append(Lx)
		iL.append(iLx)
	#####Calculate kernel values.
	#####Below we provide a naive implementation using for-loops.
	#####Parallelize this part according to your specific computing enviroment to utilize multiple GPUs.
	H = np.zeros((N, N), dtype = np.float32)
	for i in range(N):
		for j in range(N_train):
			# print(i, j)
			H[i][j] = xz(X[i], X[j], L[i], L[j], iL[i], iL[j])
	#####
	#Solve kernel regression.
	Y_train = np.ones((N_train, 10)) * -0.1
	for i in range(N_train):
		Y_train[i][y_train[i]] = 0.9
	u = H[N_train:, :N_train].dot(scipy.linalg.solve(H[:N_train, :N_train], Y_train))
	print("Trail: %d, number of layers: %d, Number of training samples: %d, test accuracy: %.4f"%(trial_i, args.depth, args.num_train, 1.0 * np.sum(np.argmax(u, axis = 1) == y_test) / N_test))
	print("Used time: %.4f"%(time()-begin_time))


# def load_cifar2(path = "../../../datasets/cifar-10-batches-py"):
# 	train_batches = []
# 	train_labels = []
# 	for i in range(1, 6):
# 		cifar_out = pickle.load(open(os.path.join(path, "data_batch_{0}".format(i)), 'rb'), encoding='bytes')
# 		train_batches.append(cifar_out[b"data"])
# 		train_labels.extend(cifar_out[b"labels"])
# 	X_train= np.vstack(tuple(train_batches)).reshape(-1, 3, 32, 32)
# 	y_train = np.array(train_labels)
# 	cifar_out = pickle.load(open(os.path.join(path, "test_batch"), 'rb'), encoding='bytes')
# 	X_test = cifar_out[b"data"].reshape(-1, 3, 32, 32)
# 	y_test = cifar_out[b"labels"]
# 	return (X_train, np.array(y_train)), (X_test, np.array(y_test))


# (X_train2, y_train2), (X_test2, y_test2) = load_cifar2('/fs/vulcan-datasets/cifar-10-python')

# os.mkdir('tmp')

# for test_id in range(32):
# 	img = Image.fromarray(X_train2[test_id].transpose(1, 2, 0))
# 	if not os.path.exists('tmp/test%d'%test_id):
# 		os.mkdir('tmp/test%d'%test_id)
# 	img.save('tmp/test%d/test%d.png'%(test_id, test_id))
# 	ids = (-H[N_train+test_id][:N_train]).argsort()[:100]
# 	for i, id in enumerate(ids):
# 		img = Image.fromarray(X_train2[id].transpose(1, 2, 0))
# 		img.save('tmp/test%d/train%d.png'%(test_id, i))