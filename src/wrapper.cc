#include "wrapper.h"
#include "fasttext.h"
#include <iostream>
#include <sstream>

#ifdef __cplusplus
extern "C" {
#endif

FastTextPtr FastText_New(void) {
  fasttext::FastText* f = new fasttext::FastText();
  // std::cerr << "New FastText instance: " << f << std::endl;
  return f;
}

void FastText_Free(FastTextPtr ptr) {
  fasttext::FastText* f = reinterpret_cast<fasttext::FastText*>(ptr);
  delete f;
}

void FastText_LoadModel(FastTextPtr ptr, char* filename) {
  fasttext::FastText* f = reinterpret_cast<fasttext::FastText*>(ptr);
  f->loadModel(filename);
  // std::cerr << "FastText LoadModel done: " << f << std::endl;
 }

void FastText_Test(FastTextPtr ptr, char* filename, int k) {
  fasttext::FastText* f = reinterpret_cast<fasttext::FastText*>(ptr);

  std::string infile = filename;
    std::ifstream ifs(infile);
    if (!ifs.is_open()) {
      std::cerr << "Test file cannot be opened!" << std::endl;
      exit(EXIT_FAILURE);
    }
    f->test(ifs, k);
    ifs.close();
}

PredictResult* FastText_Predict(FastTextPtr ptr, char* text, int k) {
  fasttext::FastText* f = reinterpret_cast<fasttext::FastText*>(ptr);
  std::vector<std::pair<float, std::string>> predictions;
  std::stringstream s(text);
  f->predict(s, k, predictions);
  PredictResult* res = new PredictResult[predictions.size() + 1];
  for (size_t i = 0; i < predictions.size(); ++i) {
    res[i].score = predictions[i].first;
    char* str = new char[predictions[i].second.size() + 1];
    strcpy(str, predictions[i].second.c_str());
    res[i].label = str;
  }
  res[predictions.size()].label = NULL;
  return res;
}

void FastText_FreeResult(PredictResult* res) {
  if (res == NULL) return;
  PredictResult* p = res;
  while (p->label) {
    delete[] p->label;
    p++;
  }
  delete[] res;
}

#ifdef __cplusplus
}
#endif 
