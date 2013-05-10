#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sstream>
#include <vector>
#include <omp.h>
#ifdef LINUX
#include <sys/time.h>
#else
#include <time.h>
#endif

using namespace std;

const int H = 55; //隐藏层 = input_size （双向的时候可以设置为两倍）
const int MAX_C = 50; //最大分类数
const int MAX_F = 300; //输入层最大的大小
const int FEATURE_SIZE = 1;
const char *model_name = "model_300_nosuff_noinit";
const bool withinit = true;

const int bptt = 3;
const int bptt_block = 10;

const char *train_file = "train.txt";
const char *valid_file = "valid.txt";
const char *test_file = "test.txt";

int class_size; //分类数
const int input_size = FEATURE_SIZE==1?55:60; //特征数，同vector_size，输入层大小

//===================== 所有要优化的参数 =====================
struct embedding_t{
	int size; //里面包含多少个变量（value 里面的变量个数） size = element_size * element_num
	int element_size; //一个向量的长度
	int element_num; //向量的个数
	double *value; //所有的参数

	void init(int element_size, int element_num){
		this->element_size = element_size;
		this->element_num = element_num;
		size = element_size * element_num;
		value = new double[size];
	}
};

embedding_t words; //词向量
embedding_t features[FEATURE_SIZE]; //除了词向量之外的其它特征

double *A; //特征矩阵：[分类数][隐藏层] 第二层的权重
double *B; //特征矩阵：[隐藏层][特征数] 第一层的权重

double *gA, *gB;

//===================== 已知数据 =====================
struct data_t{
	int word; //词的编号
	int f[FEATURE_SIZE]; //其它特征，在POS里有：1.大写；2.后缀两个词
};

typedef vector<vector<pair<data_t, int> > > dataset_t;
typedef vector<pair<data_t, int> > dataRecord_t;
//训练集
dataset_t data; //训练数据：[样本数][特征数]
int N; //训练集大小
int uN; //未知词
int *b; //目标矩阵[样本数] 训练集

//验证集
dataset_t vdata; //测试数据：[样本数][特征数]
int vN; //测试集大小
int uvN; //未知词
int *vb; //目标矩阵[样本数] 测试集

//测试集
dataset_t tdata; //测试数据：[样本数][特征数]
int tN; //测试集大小
int utN; //未知词
int *tb; //目标矩阵[样本数] 测试集


#include "fileutil.hpp"


double time_start;
double lambda = 0.0;//0.00000001; //正则项参数权重
double alpha = 0.001; //学习速率
int iter = 0;

const int thread_num = 4;
const int patch_size = thread_num;

double getTime(){
#ifdef LINUX
	timeval tv;
	gettimeofday(&tv, 0);
	return tv.tv_sec + tv.tv_usec * 1e-6;
#else
	return 1.0 * clock() / CLOCKS_PER_SEC;
#endif
}

double nextDouble(){
	return rand() / (RAND_MAX + 1.0);
}

void softmax(double hoSums[], double result[], int n){
	double max = hoSums[0];
	for (int i = 0; i < n; ++i)
		if (hoSums[i] > max) max = hoSums[i];
	double scale = 0.0;
	for (int i = 0; i < n; ++i)
		scale += exp(hoSums[i] - max);
	for (int i = 0; i < n; ++i)
		result[i] = exp(hoSums[i] - max) / scale;
}

double sigmoid(double x){
	return 1 / (1 + exp(-x));
}

//b = Ax
void fastmult(double *A, double *x, double *b, int xlen, int blen){
	double val1, val2, val3, val4;
	double val5, val6, val7, val8;
	int i;
	for (i=0; i<blen/8*8; i+=8) {
		val1=0;
		val2=0;
		val3=0;
		val4=0;

		val5=0;
		val6=0;
		val7=0;
		val8=0;

		for (int j=0; j<xlen; j++) {
			val1 += x[j] * A[j+(i+0)*xlen];
			val2 += x[j] * A[j+(i+1)*xlen];
			val3 += x[j] * A[j+(i+2)*xlen];
			val4 += x[j] * A[j+(i+3)*xlen];

			val5 += x[j] * A[j+(i+4)*xlen];
			val6 += x[j] * A[j+(i+5)*xlen];
			val7 += x[j] * A[j+(i+6)*xlen];
			val8 += x[j] * A[j+(i+7)*xlen];
		}
		b[i+0] += val1;
		b[i+1] += val2;
		b[i+2] += val3;
		b[i+3] += val4;

		b[i+4] += val5;
		b[i+5] += val6;
		b[i+6] += val7;
		b[i+7] += val8;
	}

	for (; i<blen; i++) {
		for (int j=0; j<xlen; j++) {
			b[i] += x[j] * A[j+i*xlen];
		}
	}
}

void setInputVector(data_t *id, double *x){
	int j = 0;
	int offset = id->word * words.element_size;
	for(int k = 0; k < words.element_size; k++,j++){
		x[j] = words.value[offset + k];
	}

	for(int f = 0; f < FEATURE_SIZE; f++){
		embedding_t &em = features[f];
		offset = id->f[f] * em.element_size;
		for(int k = 0; k < em.element_size; k++,j++){
			x[j] = em.value[offset + k];
		}
	}
}

void updateInputVector(data_t *id, double *dx){
	int offset;
	int j = 0;

	offset = id->word * words.element_size;
	for(int k = 0; k < words.element_size; k++,j++){
		int t = offset + k;
		words.value[t] += alpha * (dx[j] - lambda * words.value[t]);
	}

	for(int f = 0; f < FEATURE_SIZE; f++){
		embedding_t &em = features[f];
		offset = id->f[f] * em.element_size;
		for(int k = 0; k < em.element_size; k++,j++){
			int t = offset + k;
			em.value[t] += alpha * (dx[j] - lambda * em.value[t]);
		}
	}
}


double checkCase(data_t *id, double *state, double *nextState, int ans, int &correct, bool gd = false, double *dState = NULL){
	double x[MAX_F];
	setInputVector(id, x);

	double h[H] = {0};
	fastmult(B, state, h, input_size, H);
	for(int i = 0; i < H; i++){
		nextState[i] = h[i] = sigmoid(h[i] + x[i]);
	}
	//for(int i = 0, k=0; i < H; i++){
	//	for(int j = 0; j < input_size; j++,k++){
	//		//h[i] += x[j] * B[i*input_size+j];
	//		h[i] += x[j] * B[k];
	//	}
	//	h[i] = sigmoid(h[i]);
	//}

	double r[MAX_C] = {0};
	for(int i = 0; i < class_size; i++){
		for(int j = 0; j < H; j++){
			r[i] += h[j] * A[i*H+j];
		}
	}
	double y[MAX_C];
	softmax(r, y, class_size);

	if(gd){ //修改参数

		//矩阵 A 直接更新
		for(int i = 0; i < class_size; i++){
			double v = (i==ans?1:0) - y[i];
			for(int j = 0; j < H; j++){
				int t = i*H+j;
				A[t] += alpha * (v * h[j] - lambda * A[t]);
			}
		}

		//隐含层的变化量
		double dh[H] = {0};
		for(int j = 0; j < H; j++){
			dh[j] = A[ans*H+j];
			for(int i = 0; i < class_size; i++){
				dh[j] -= y[i]*A[i*H+j];
			}
		}

		//保存这次的隐藏层变化量
		for(int j = 0; j < H; j++){
			dState[j] = dh[j];
		}
	}

	bool ok = true;
	for(int i = 0; i < class_size; i++){
		if(i != ans && y[i] >= y[ans])
			ok = false;
	}

	if(ok)
		correct++;
	return log(y[ans]); //计算似然
}


void BPTT(vector<vector<double> > &states, vector<vector<double> > &dStates, vector<data_t> &data){
	int len = (int)states.size();

	double dh[H];
	//初始化隐藏层
	for(int i = 0; i < H; i++){
		dh[i] = dStates[len-1][i];
		dStates[len-1][i] = 0; //梯度只用一次（可以像rnnlm考虑复用试试）
	}
	//终止条件：不走到初始层；bptt的总次数符合预设（在句子结尾处可能会多迭代几层，不影响结果）
	for(int pos = len-1,cnt=0; pos>0 && cnt<bptt+bptt_block-1; pos--,cnt++){ //边界一会再确定，限制：边缘、步数
		vector<double> &h = states[pos];

		//早一层
		vector<double> &state = states[pos-1];
		vector<double> &ds = dStates[pos-1];

		//隐含层变化量求导
		for(int i = 0; i < H; i++){
			dh[i] *= h[i]*(1-h[i]);
		}

		data_t *id = &data[pos-1];
		updateInputVector(id, dh);

		double dx[MAX_F] = {0};
		for(int i = 0; i < H; i++){
			for(int j = 0; j < input_size; j++){
				int t = i*input_size+j;
				dx[j] += dh[i] * B[t]; //这里可以优化成矩阵乘法
				gB[t] += state[j] * dh[i];
			}
		}

		for(int i = 0; i < H; i++){
			dh[i] = dx[i] + ds[i];
			ds[i] = 0; //梯度只用一次（可以像rnnlm考虑复用试试）
		}
	}

	for(int i = 0; i < H; i++){
		for(int j = 0; j < input_size; j++){
			int t = i*input_size+j;
			B[t] += alpha * (gB[t] - lambda * B[t]);
			gB[t] = 0;
		}
	}

}

void writeFile(const char *name, double *A, int size){
	FILE *fout = fopen(name, "wb");
	fwrite(A, sizeof(double), size, fout);
	fclose(fout);
}


double checkSet(dataset_t &data, int &correct, int &correctU){
	correct = 0;
	double ret = 0;
	for(size_t i = 0; i < data.size(); i++){
		double _state[H] = {0}, _state2[H];
		for(int j = 0; j < H; j++) _state[j] = 0.1;
		double *state = _state;
		double *nextState = _state2;
		for(size_t j = 0; j < data[i].size(); j++){
			//for(int s = 0; s < N; s++){
			int tc = 0;
			double tv = checkCase(&data[i][j].first, state, nextState, data[i][j].second, tc);

			ret += tv;
			correct += tc;
			if(data[i][j].first.word == 1739){
				correctU += tc;
			}
			swap(state, nextState);
		}
	}
	return ret;
}

//检查正确率和似然
//返回值是似然
double check(){
	double ret = 0, ev, et;
	int correct = 0, correctTest = 0, correctValid = 0;
	int correctU = 0, correctTestU = 0, correctValidU = 0;

	ret = checkSet(data, correct, correctU);
	ev = checkSet(vdata, correctValid, correctValidU);
	et = checkSet(tdata, correctTest, correctTestU);

	double ps = 0;
	int pnum = 0;
	for(int i = 0; i < class_size*H; i++,pnum++){
		ps += A[i]*A[i];
	}
	for(int i = 0; i < H*input_size; i++,pnum++){
		ps += B[i]*B[i];
	}
	for(int i = 0; i < words.size; i++,pnum++){
		ps += words.value[i]*words.value[i];
	}
	for(int k = 0; k < FEATURE_SIZE; k++){
		for(int i = 0; i < features[k].size; i++,pnum++){
			ps += features[k].value[i]*features[k].value[i];
		}
	}

	char fname[100];
	sprintf(fname, "%s_A", model_name);
	writeFile(fname, A, class_size*H);
	sprintf(fname, "%s_B", model_name);
	writeFile(fname, B, H*input_size);
	sprintf(fname, "%s_w", model_name);
	writeFile(fname, words.value, words.size);
	//特征等要的时候再存
	//sprintf(fname, "%s_f1", model_name);
	//writeFile(fname, features[1].value, features[1].size);

	double fret = -ret/N + ps/pnum*lambda/2;
	printf("train: %lf %d/%d(%.2lf%%,%.2lf%%), valid: %lf %d/%d(%.2lf%%,%.2lf%%), test: %lf %d/%d(%.2lf%%,%.2lf%%) time:%.1lf\n",
		-ret/N, correct, N, 100.*correct/N, 100.*correctU/uN,
		-ev/vN, correctValid, vN, 100.*correctValid/vN, 100.*correctValidU/uvN,
		-et/tN, correctTest, tN, 100.*correctTest/tN, 100.*correctTestU/utN,
		getTime()-time_start);
	fflush(stdout);
	return fret;
}


int readFile(const char *name){
	FILE *fin = fopen(name, "rb");
	if(!fin)
		return 0;
	size_t t = fread(A, 1, sizeof(A), fin);
	fclose(fin);
	return 1;
}

int main(){
	printf("read data size\n");

	class_size = 45;

	init(train_file);

	words.init(50, 130000);
	features[0].init(5, 5);
	if(FEATURE_SIZE > 1)
		features[1].init(5, 455);

	printf("read data\n");
	readAllData(train_file, "Train", data, N, uN);
	readAllData(valid_file, "Valid", vdata, vN, uvN);
	readAllData(test_file, "Test", tdata, tN, utN);

	printf("init. input(features):%d, hidden:%d, output(classes):%d, alpha:%lf, lambda:%.16lf\n", input_size, H, class_size, alpha, lambda);
	printf("vocab_size:%d\n", words.element_num);

	A = new double[class_size*H];
	gA = new double[class_size*H];
	B = new double[H*input_size];
	gB = new double[H*input_size];

	//if(!readFile("model_gd")){
	for(int i = 0; i < class_size * H; i++){
		A[i] = nextDouble()-0.5;
	}
	for(int i = 0; i < H * input_size; i++){
		B[i] = nextDouble()-0.5;
		gB[i] = 0;
	}
	for(int i = 0; i < words.size; i++){
		words.value[i] = nextDouble()-0.5;
	}
	for(int k = 0; k < FEATURE_SIZE; k++){
		for(int i = 0; i < features[k].size; i++){
			features[k].value[i] = nextDouble()-0.5;
		}
	}

	if(withinit){
		for(int i = 0; i < words.element_num; i++){
			for(int j = 0; j < words.element_size; j++){
				words.value[i * words.element_size + j] = senna_raw_words[i].vec[j];
			}
		}
	}


	time_start = getTime();

	int *order = new int[data.size()];
	for(int i = 0; i < data.size(); i++){
		order[i] = i;
	}

	double lastLH = 1e100;
	while(1){
		//计算正确率
		printf("iter: %d, ", iter++);
		double LH = check();
		/*if(LH > lastLH){
		alpha = 0.0001;
		}
		lastLH = LH;*/

		int cnt = 0;

		double lastTime = getTime();
		//memset(gA, 0, sizeof(double)*class_size*H);
		//memset(gB, 0, sizeof(double)*H*input_size);

		for(size_t i = 0; i < data.size(); i++){
			swap(order[i], order[rand()%data.size()]);
		}
		for(size_t i = 0; i < data.size(); i++){
			int j = 0;
			int s = order[i+j];
			dataRecord_t &dr = data[s];

			vector<double> state(H);

			//BPTT使用的变量
			vector<vector<double> > bpStates;
			vector<vector<double> > bpDStates;
			vector<data_t> bpData;

			for(int j = 0; j < H; j++) state[j] = 0.1;
			for(size_t j = 0; j < dr.size(); j++){
				vector<double> nextState(H);
				vector<double> dState(H);
				int tc = 0;
				checkCase(&dr[j].first, &state[0], &nextState[0], dr[j].second, tc, true, &dState[0]);
				bpData.push_back(dr[j].first);
				bpDStates.push_back(dState);
				bpStates.push_back(nextState);

				if(cnt % bptt_block == 0 || j == dr.size()-1) //判断执行bptt
					BPTT(bpStates, bpDStates, bpData);

				state = nextState;
				cnt++;
				if ((cnt%1000)==0){
					//	printf("%cIter: %3d\t   Progress: %.2f%%   Words/sec: %.1f ", 13, iter, 100.*cnt/N, cnt/(getTime()-lastTime));
				}
			}


		}
		//for(int i = 0; i < vN; i++){
		//	int s = i;
		//	data_t *x = vdata + s * window_size;
		//	int ans = vb[s];
		//	int tmp;
		//	checkCase(x, ans, tmp, true);

		//	if ((i%100)==0){
		//	//	printf("%cIter: %3d\t   Progress: %.2f%%   Words/sec: %.1f ", 13, iter, 100.*i/N, i/(getTime()-lastTime));
		//	}
		//}
		//printf("%c", 13);
	}
	return 0;
}
