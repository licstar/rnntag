extern "C"
{
#include "SENNA_Tokenizer.h"
}
#include <string>
#include <map>
using namespace std;

#define MAX_STRING 100

SENNA_Tokenizer *tokenizer; //分词器

struct senna_node{
	char str[102];
	double vec[50];
}senna_raw_words[130000]; //单词的原文及训练好的向量

void initWithSenna(){
	FILE *fvec = fopen("embeddings/embeddings.txt", "r");
	FILE *fword = fopen("hash/words.lst", "r");
	char line[1024];
	
	//读取数据
	int n = 0;
	while(fgets(line, sizeof(line), fvec)){
		char *token;
		for(int i = 0; i < 50; i++){
			token = strtok(i?NULL:line, " \n");
			senna_raw_words[n].vec[i] = atof(token);
		}
		fgets(line, sizeof(line), fword);
		line[strlen(line)-1] = 0; //去除最后的\n
		strcpy(senna_raw_words[n].str, line);
		n++;
	}
	fclose(fvec);
	fclose(fword);
}

SENNA_Tokenizer* initTokenizer(){
	char *opt_path = NULL;
	    /* inputs */
    SENNA_Hash *word_hash = SENNA_Hash_new(opt_path, "hash/words.lst");
    SENNA_Hash *caps_hash = SENNA_Hash_new(opt_path, "hash/caps.lst");
    SENNA_Hash *suff_hash = SENNA_Hash_new(opt_path, "hash/suffix.lst");
    SENNA_Hash *gazt_hash = SENNA_Hash_new(opt_path, "hash/gazetteer.lst");

    SENNA_Hash *gazl_hash = SENNA_Hash_new_with_admissible_keys(opt_path, "hash/ner.loc.lst", "data/ner.loc.dat");
    SENNA_Hash *gazm_hash = SENNA_Hash_new_with_admissible_keys(opt_path, "hash/ner.msc.lst", "data/ner.msc.dat");
    SENNA_Hash *gazo_hash = SENNA_Hash_new_with_admissible_keys(opt_path, "hash/ner.org.lst", "data/ner.org.dat");
    SENNA_Hash *gazp_hash = SENNA_Hash_new_with_admissible_keys(opt_path, "hash/ner.per.lst", "data/ner.per.dat");

	int opt_usrtokens = 1;

    SENNA_Tokenizer *tokenizer = SENNA_Tokenizer_new(word_hash, caps_hash, suff_hash, gazt_hash, gazl_hash, gazm_hash, gazo_hash, gazp_hash, opt_usrtokens);
	return tokenizer;
}


void readWord(char *word, FILE *fin){
	int a=0, ch;

	while (!feof(fin)) {
		ch=fgetc(fin);

		if (ch==13) continue;

		if ((ch==' ') || (ch=='\t') || (ch=='\n')) {
			if (a>0) {
				if (ch=='\n') ungetc(ch, fin);
				break;
			}

			if (ch=='\n') {
				strcpy(word, (char *)"</s>");
				return;
			}
			else continue;
		}

		word[a]=ch;
		a++;

		if (a>=MAX_STRING) {
			printf("Too long word found!\n");   //truncate too long words
			a--;
		}
	}
	word[a]=0;
}

data_t simplifyWord(char *word){
	char tword[100];
	strcpy(tword, word);
	int len = strlen(tword);
	tword[len] = ' ';
	tword[len+1] = 0;
	SENNA_Tokens* tokens = SENNA_Tokenizer_tokenize(tokenizer, tword);

	data_t ret;

	if(tokens->n == 0){
		strcpy(word, "UNKNOWN");
		ret.word = 1739;
		ret.f[0] = 0;
		if(FEATURE_SIZE > 1) ret.f[1] = 48;
		printf("token to empty!\n");
	}else if(tokens->n > 1){
		strcpy(word, "UNKNOWN");
		ret.word = 1739;
		ret.f[0] = 0;
		if(FEATURE_SIZE > 1) ret.f[1] = 48;
	}else{
		ret.word = tokens->word_idx[0];
		ret.f[0] = tokens->caps_idx[0];
		if(FEATURE_SIZE > 1) ret.f[1] = tokens->suff_idx[0];
	}
	return ret;
}

data_t readWordIndex(FILE *fin, int &tag){
	char word[MAX_STRING];
	data_t ret;
	ret.word = 0;

	readWord(word, fin);
	if (feof(fin)) return ret;

	tag = 0;
	if(strcmp("</s>", word) != 0){
		for(int k = strlen(word)-1; k >=0; k--){
			if(word[k] == '/'){
				tag = atoi(word+k+1);
				word[k] = 0;
				break;
			}
		}
		ret = simplifyWord(word);
	}

	return ret;
}

void init(const char *train_file){
	initWithSenna(); //senna的字典
	tokenizer = initTokenizer(); //分词器

	//learnVocab(train_file);
//	for(int i = 0; i < 130000; i++){
//		addWord(senna_raw_words[i].str);
//	}
	//readalldata的时候，需要存储别的特征，caps和suff

}


void readAllData(const char *file, const char *dataset, dataset_t &mydata, int &N, int &uN){
	FILE *fi=fopen(file, "rb");
	
	vector<pair<data_t, int> > line;

	data_t padding; //这个想办法初始化一下
	padding.word = 1738;
	padding.f[0] = 0;
	if(FEATURE_SIZE > 1) padding.f[1] = 48;


	N = 0;
	while(1){
		int tag;
		data_t dt = readWordIndex(fi, tag);
		if (feof(fi)) break;
		line.push_back(make_pair(dt, tag));

		if(dt.word == 0){
			line.pop_back();
			mydata.push_back(line);
			N += line.size();
			line.clear();
		}
	}
	fclose(fi);

	int unknown = 0;
	for(size_t i = 0, offset=0; i < mydata.size(); i++){
		vector<pair<data_t, int> > &vec = mydata[i];
		for(int j = 0; j < (int)vec.size(); j++){
			if(vec[j].first.word == 1739)
				unknown++;
		}
	}

	printf("%s data: N(words):%d, unknown:%d\n", dataset, N, unknown);
	uN = unknown;
}
