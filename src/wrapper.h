#ifndef _FAST_TEXT_WRAPPER_
#define _FAST_TEXT_WRAPPER_

#ifdef __cplusplus
extern "C" {
#endif

typedef void* FastTextPtr;

typedef struct {
  float score;
  char* label;
} PredictResult;

FastTextPtr FastText_New(void);
void FastText_Free(FastTextPtr);

void FastText_LoadModel(FastTextPtr, char* filename);

void FastText_Test(FastTextPtr, char* filename, int k);

PredictResult* FastText_Predict(FastTextPtr, char* text, int k);

void FastText_FreeResult(PredictResult* res);

#ifdef __cplusplus
}
#endif 

#endif

