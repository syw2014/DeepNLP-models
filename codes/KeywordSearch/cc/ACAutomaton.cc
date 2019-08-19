// An implementation of Aho-Corasick String search algorithm
// Author: yw.shi
// Date: 2019-08-06 
#include <map>
#include <string>
#include <vector>
#include <queue>
#include <iostream>
#include <fstream>
#include <sstream>
#include <time.h>
#include <io.h>

#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include<ctype.h>

#ifdef WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif
#include <dirent.h>
using namespace std;

// encode transform for chinese character
wchar_t* MBCS2Unicode(wchar_t * buff, const char * str)
{
	wchar_t * wp = buff;
	char * p = (char *)str;
	while (*p)
	{
		if (*p & 0x80)
		{
			*wp = *(wchar_t *)p;
			p++;
		}
		else {
			*wp = (wchar_t)*p;
		}
		wp++;
		p++;
	}
	*wp = 0x0000;
	return buff;
}


char * Unicode2MBCS(char * buff, const wchar_t * str)
{
	wchar_t * wp = (wchar_t *)str;
	char * p = buff, *tmp;
	while (*wp) {
		tmp = (char *)wp;
		if (*wp & 0xFF00) {
			*p = *tmp;
			p++; tmp++;
			*p = *tmp;
			p++;
		}
		else {
			*p = *tmp;
			p++;
		}
		wp++;
	}
	*p = 0x00;
	return buff;
}

// convert string to wstring
wstring str2wstr(string str) {
	size_t len = str.size();
	wchar_t* b = (wchar_t *)malloc((len + 1) * sizeof(wchar_t));
	MBCS2Unicode(b, str.c_str());
	wstring r(b);
	free(b);
	return r;
}

// wstring to string
string wstr2str(wstring wstr) {
	size_t len = wstr.size();
	char * b = (char *)malloc((2 * len + 1) * sizeof(char));
	Unicode2MBCS(b, wstr.c_str());
	string r(b);
	free(b);
	return r;
}
//---------------------------

class TrieNode {
public:
	string word;	// token as word
	map<string, TrieNode*> next;	// next node, TODO, can be optimized with DobueArrayTree
	TrieNode *fail;			// failure pointer
	bool isMatched;			// whether matched, match means root node
	int termFreq;			// frequency of matched word
	int wordLength;			// word number in string 

	int index;				// first position of occur in string

	// TODO, add pattern label
	int label;				// pattern label,

public:
	TrieNode() : word(""), fail(0), isMatched(false), termFreq(0), wordLength(0), index(-1), label(-1) {};	// construct and initialize

};

class ACAutomaton {
public:
	TrieNode *root;			// root node
	vector<TrieNode*> instances;		// store all tail nodes

public:
	// constuction
	ACAutomaton() {
		root = new TrieNode;
	}
		
	// de-constuction
	~ACAutomaton() {
		delete root;
	}

	/*
	* split string into word btyes, record word length and characters
	*/
	void splitWord(const string &text, int &wordlength, vector<string> &characters) {
		// convert string to wstring
		int wordSize = text.size();	// get number of bytes
		/*int i = 0;
		while (i < wordSize) {
			int characterSize = 1;

			if (text[i] & 0x80) {
				char character = text[i];
				character <<= 1;
				do {
					character <<= 1;
					++characterSize;
				} while (character & 0x80);
			}

			string subWord;
			subWord = text.substr(i, characterSize);
			characters.push_back(subWord);

			i += characterSize;
			++wordlength;
		}
		*/
		// ref: https://en.wikipedia.org/wiki/UTF-8#Description
		for (int i = 0; i < wordSize;) {	
			int cplen = 1;
			if ((text[i] & 0x80) == 0xf0) {			// 11111000, 11110000
				cplen = 4;
			} else if ((text[i] & 0xf0) == 0xe0) {	// 11100000
				cplen = 3;
			} else if ((text[i] & 0xe0) == 0xc0) {	// 11000000
				cplen = 2;
			}
			if ((i + cplen) > wordSize) {			// last word
				cplen = 1;
			}
			characters.push_back(text.substr(i, cplen));
			i += cplen;
			++wordlength;
		}
		
	}

	// Get Next node of the currenct node
	TrieNode* getNext(TrieNode *current, string &character) {

		map<string, TrieNode*>::iterator iter = current->next.find(character);	// get iters
		if (iter != current->next.end()) {
			return iter->second;	// if found return trie Node pointer
		}

		// check whether current node was root
		if (current == root) {
			return root;
		}

		return 0;
	}

	// Add word into trie tree
	void add(const string &pattern, const int &label) {
		int wordLength = 0;
		vector<string> characters;
		splitWord(pattern, wordLength, characters);
		//cout << "add test: " << pattern << " number of word: " << wordLength << endl;

		TrieNode *tmp = root;

		int i = 1;		// word length counter
		string pathWord = "";
		for (vector<string>::iterator iter = characters.begin(); iter != characters.end(); ++iter) {
			pathWord.append(*iter);
			map<string, TrieNode*>::iterator item = tmp->next.find(*iter);	// check strings in node path
			if (item != tmp->next.end()) {
				tmp = item->second;		// if exist return trie node
			}
			else {						// if not exist then create new node in the graph or tree
				TrieNode *n = new TrieNode;
				n->word = pathWord;	
				n->wordLength = i;
				tmp->next.insert(make_pair(*iter, n));
				tmp = n;
			}
			
			if (iter + 1 == characters.end()) {
				tmp->isMatched = true;
				// cout << "full word: " << tmp->word << endl;
			}
			++i;
		}
		tmp->label = label;		// pattern label
	}

	// create dictionary
	void build() {

		queue<TrieNode*> que;	// 

		// set all fail pointer as root of the node which depth=1, and add all the nodes into queue
		for (map<string, TrieNode*>::iterator iter = root->next.begin(); iter != root->next.end(); ++iter) {
			iter->second->fail = root;
			que.push(iter->second);
		}

		instances.clear();
		//instances.push_back(root);

		while (!que.empty()) {
			TrieNode* temp = que.front();
			que.pop();

			// loop the next node of current node
			for (map<string, TrieNode*>::iterator iter = temp->next.begin(); iter != temp->next.end(); ++iter) {
				string character = iter->first;	// get the string

				// push the current node into queue
				que.push(iter->second);

				// set fail pointer, search from parent node
				TrieNode  *parent = temp->fail;
				while (!getNext(parent, character)) parent = parent->fail;
				iter->second->fail = getNext(parent, character);
				if (!getNext(parent, character)) throw 1;
			}

			// store tail node
			if (temp->isMatched) {
				instances.push_back(temp);
			}
		}
	}

	// match pattern in given text, match succeed return true
	bool match(const string &text) {
		int bufLength = 0;
		vector<string> characters;
		splitWord(text, bufLength, characters);

		TrieNode *tmp = root;
		for (vector<string>::iterator iter = characters.begin(); iter != characters.end(); ++iter) {
			while (!getNext(tmp, *iter)) {
				tmp = tmp->fail;
			}
			tmp = getNext(tmp, *iter);
			if (tmp->isMatched) {
				return true;
			}
		}
		return false;
	}

	// return all the matched nodes
	void search(const string &text, map<string, TrieNode*> &nodes) {
		int bufLength = 0;
		vector<string> characters;
		splitWord(text, bufLength, characters);

		int index = 0;

		TrieNode *temp = root;
		 
		for (vector<string>::iterator character = characters.begin(); character != characters.end(); ++character) {

			while (!getNext(temp, *character)) {
				temp = temp->fail;
			}

			temp = getNext(temp, *character);

			if (temp->isMatched) { // match
				map<string, TrieNode*>::iterator nodeFind = nodes.find(temp->word);
				if (nodeFind == nodes.end()) {
					temp->termFreq = 1;
					temp->index = index + 1 - temp->wordLength;
					nodes.insert(make_pair(temp->word, temp));
				}
				else {
					nodeFind->second->termFreq += 1;
				}
			}
			++index;
		}
	}

}; // end ACAutomaton class

// replace string characters
void string_replace(string &strBig, const string &strsrc, const string &strdst)
{
	string::size_type pos = 0;
	string::size_type srclen = strsrc.size();
	string::size_type dstlen = strdst.size();

	while ((pos = strBig.find(strsrc, pos)) != string::npos)
	{
		strBig.replace(pos, srclen, strdst);
		pos += dstlen;
	}
}

// search pattern in given doc
void search(string &text, ACAutomaton* ahc, vector<string> &keywords, vector<int> &position, vector<int> &labels) {
	
	// clear results
	keywords.clear();
	position.clear();
	labels.clear();

	// firstly check whether can match pattern in text, if match then get match patterns(keywords)
	bool isMatched = ahc->match(text);

	if (isMatched == true) {
		map<string, TrieNode*> nodes;
		ahc->search(text, nodes);

		// parse search results
		for (map<string, TrieNode*>::iterator iter = nodes.begin(); iter != nodes.end(); ++iter) {
			keywords.push_back(iter->second->word);		// get pattern(keyword)
			position.push_back(iter->second->index);	// position of first appearance
			labels.push_back(iter->second->label);		// label of keyword
		}
	}
}

// search patterns in given file, each line as one doc
void searchInFile(const string inFile, ACAutomaton* ahc, const string outFile) {
	// open file and process
	ifstream fin(inFile.c_str());

	if (!fin.is_open()) {
		perror(("Error while opening file " + inFile).c_str());
	}

	// open output file
	ofstream fout(outFile.c_str());
	if (!fout.is_open()) {
		perror(("Error while opening output file " + outFile).c_str());
	}

	// process
	string line;
	vector<string> keywords;
	vector<int> position;
	vector<int> labels;
	while (getline(fin, line)) {
		search(line, ahc, keywords, position, labels);

		// TODO, write to file
		vector<string>::iterator wordIter = keywords.begin();
		vector<int>::iterator posIter = position.begin();
		vector<int>::iterator labelIter = labels.begin();
		string str = "";
		stringstream ss;
		for (; wordIter != keywords.end() && posIter != position.end() && labelIter != labels.end(); 
			++wordIter, ++posIter, ++labelIter) {
			
			str += *wordIter;
			ss << *posIter;
			str += "," + ss.str();
			ss.str("");		// clear string stream
			ss << *labelIter;
			str += "," + ss.str() + "\t";
			ss.str("");
		}
		
		// remove last tab
		str = str.substr(0, str.length() - 1);
		fout << line + "\t" + str << endl;
	}
	
	fin.close();
	fout.close();
}


// get all files in given directory
void getFiles(string path, vector<string>& files) {
	DIR *pDir;
	struct dirent* ptr;
	if (!(pDir = opendir(path.c_str()))) {
		perror(("Folder " + path + "doesn't exist!").c_str());
		return;
	}
	while ((ptr = readdir(pDir)) != 0) {
		if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
			cout << ptr->d_name << endl;
			files.push_back(path + "/" + ptr->d_name);
		}
	}
	closedir(pDir);
}

// string split with pattern
void split(const string &str, vector<string>& vec, const string& pattern) {
	string::size_type pos1, pos2;
	pos1 = 0;
	pos2 = str.find(pattern);
	while (string::npos != pos2) {
		vec.push_back(str.substr(pos1, pos2 - pos1));
		pos1 = pos2 + pattern.size();
		pos2 = str.find(pattern, pos1);
	}
	if (pos1 != str.length()) {
		vec.push_back(str.substr(pos1));
	}
}

// parameters parser
int ArgPos(char *str, int argc, char **argv) {
	int a;
	for (a = 1; a < argc; ++a) if (!strcmp(str, argv[a])) {
		if (a == argc - 1) {
			printf("Argument missing for %s\n", str);
			exit(1);
		}
		return a;
	}
	return -1;
}

void usage() {
	printf("Usage:\n");
	printf("\t ./exec-file -dict_dir [input dictionary path] -input [input document filename] -output [output match result filename]\n");
}


int main(int argc, char* argv[]) {

	if (argc < 3) {
		usage();
		return 0;
	}
	
	
	ACAutomaton *ahc = new ACAutomaton; // create instance


	string infile = "";		// input documents
	string outfile = "";	// output result file, line \t keyword,position,label
	string dictDir = "";	// path of input dictionary
	int i;	// position of parameters
	if ((i = ArgPos((char*)"-dict_dir", argc, argv)) > 0) dictDir = argv[i + 1];
	if ((i = ArgPos((char *)"-input", argc, argv)) > 0) infile = argv[i + 1];	
	if ((i = ArgPos((char *)"-output", argc, argv)) > 0) outfile = argv[i + 1];

	// Load keywords to create trie tree
	// get files from dictionary directory
	vector<string> dictFiles;
	getFiles(dictDir, dictFiles);
	if (dictFiles.size() == 0) {
		perror(("Found none files in input dictionary path!" + dictDir).c_str());
	}
	ifstream fin;
	string word;
	vector<string> arr;
	int counts = 0;
	int label = -1;
	clock_t starts, ends;
	stringstream ss;
	starts = clock();	// time start
	for (vector<string>::iterator iter = dictFiles.begin(); iter != dictFiles.end(); ++iter) {
		// parse keyword label
		arr.clear();
		split(*iter, arr, "_");
		// extract file label from file name, here we length of label id was 3.
		string name = (arr[0]).substr((arr[0]).length() - 3);
		ss.str("");
		ss << name;
		ss >> label;
		
		fin.open((*iter).c_str());
		if (fin) {
			while (getline(fin, word)) {
				// parse line
				arr.clear();
				split(word, arr, "\t");
				ahc->add(arr[0], label);
				++counts;
			}
		}
		else {
			cout << "Open file " << *iter << "failed !" << endl;
		}
		fin.clear();
		fin.close();
	}
	ends = clock();
	cout << "Load dictionary from path<" << dictDir << "> total words: "  << counts << " times: "
		<< (float)(ends - starts) / CLOCKS_PER_SEC << "ms" << endl;

	// build trie tree
	starts = clock();
	ahc->build();
	ends = clock();
	cout << "Build trie tree completed cost " << (float)(ends - starts) / CLOCKS_PER_SEC << "ms" << endl;

	// Open file to process line
	starts = clock();
	searchInFile(infile, ahc, outfile);
	ends = clock();
	cout << "Search text complete cost " << (float)(ends - starts) / CLOCKS_PER_SEC << "ms" << endl;


	delete ahc;

	return 0;
}


