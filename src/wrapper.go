package fasttext

// #cgo CXXFLAGS: -pthread -std=c++0x
// #include "wrapper.h"
// #include <stdlib.h>
// PredictResult* PredictResultN(PredictResult*p, int n) { return p + n; }
import "C"
import "unsafe"
import "strings"

type FastText struct {
	ptr C.FastTextPtr
}

type Result struct {
	Score float32
	Label string
}

func New(modelFile string) FastText {
	f := FastText{C.FastText_New()}
	cfile := C.CString(modelFile)
	C.FastText_LoadModel(f.ptr, cfile)
	C.free(unsafe.Pointer(cfile))
	return f
}

func (f FastText) Free() {
	C.FastText_Free(f.ptr)
	// f.ptr = C.FastTextPtr(0)
}

func (f FastText) Test(filename string, k int) {
	cfile := C.CString(filename)
	C.FastText_Test(f.ptr, cfile, C.int(k))
	C.free(unsafe.Pointer(cfile))
}

func (f FastText) Predict(text string, k int) []Result {
	cstr := C.CString(text)
	defer C.free(unsafe.Pointer(cstr))

	p := C.FastText_Predict(f.ptr, cstr, C.int(k))
	defer C.FastText_FreeResult(p)

	v := make([]Result, 0)
	for i := 0; i < k; i++ {
		cres := C.PredictResultN(p, C.int(i))
		label := C.GoString(cres.label)
		if len(label) == 0 {
			break
		}
		if strings.HasPrefix(label, "__label__") {
			label = label[len("__label__"):]
		}
		v = append(v, Result{float32(cres.score), label})
	}
	return v
}