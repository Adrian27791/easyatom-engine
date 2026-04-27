// =============================================================================
// EasyAtom · Ladrillo 8 — Bindings JNI para Android.
// =============================================================================
//
// Este archivo es OPCIONAL: solo se compila como parte del .so de Android
// (ver android/app/src/main/cpp/CMakeLists.txt). NO entra en el ejecutable
// de tests del motor — los tests usan el C ABI directamente.
//
// Convenciones:
//   * Paquete Java/Kotlin: com.easyhelpcare.easyatom
//   * Clase nativa:        EasyAtomNative
//   * Handle:              long que envuelve un eatom_kernel_t*.
//
// Métodos expuestos:
//   long  nativeCreate(int dim, long seed);
//   void  nativeDestroy(long handle);
//   int   nativeIngest(long handle, String name);
//   int   nativeQueryArgmax(long handle, String[] roles, String[] fillers,
//                           String queryRole, String[] candidates,
//                           boolean autoingest);
//   double[] nativeQueryProbs(...) → null en error.
//
// Ningún jthrow desde aquí: los errores fluyen como int o null.

#ifdef EASYATOM_BUILD_JNI

#include <jni.h>
#include <string>
#include <vector>

#include "easyatom/c_api.h"

namespace {

class JStringUTF {
public:
    JStringUTF(JNIEnv* env, jstring s) : env_(env), s_(s) {
        c_ = (s_ != nullptr) ? env_->GetStringUTFChars(s_, nullptr) : nullptr;
    }
    ~JStringUTF() {
        if (c_ != nullptr) env_->ReleaseStringUTFChars(s_, c_);
    }
    JStringUTF(const JStringUTF&) = delete;
    JStringUTF& operator=(const JStringUTF&) = delete;
    [[nodiscard]] const char* c_str() const noexcept { return c_; }

private:
    JNIEnv* env_;
    jstring s_;
    const char* c_;
};

class JStringArray {
public:
    JStringArray(JNIEnv* env, jobjectArray arr) : env_(env) {
        if (!arr) return;
        n_ = env->GetArrayLength(arr);
        for (jsize i = 0; i < n_; ++i) {
            auto js = (jstring)env->GetObjectArrayElement(arr, i);
            const char* c = (js ? env->GetStringUTFChars(js, nullptr) : nullptr);
            owned_.emplace_back(env, js, c);
            ptrs_.push_back(c);
        }
    }
    ~JStringArray() {
        for (auto& [env, js, c] : owned_) {
            if (c) env->ReleaseStringUTFChars(js, c);
            if (js) env->DeleteLocalRef(js);
        }
    }
    [[nodiscard]] const char* const* data() const { return ptrs_.data(); }
    [[nodiscard]] size_t size() const { return ptrs_.size(); }

private:
    JNIEnv* env_;
    jsize n_ = 0;
    std::vector<std::tuple<JNIEnv*, jstring, const char*>> owned_;
    std::vector<const char*> ptrs_;
};

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_easyhelpcare_easyatom_EasyAtomNative_nativeCreate(
    JNIEnv*, jclass, jint dim, jlong seed) {
    if (dim <= 0) return 0;
    auto* k = eatom_kernel_create(static_cast<size_t>(dim),
                                  static_cast<uint64_t>(seed));
    return reinterpret_cast<jlong>(k);
}

JNIEXPORT void JNICALL
Java_com_easyhelpcare_easyatom_EasyAtomNative_nativeDestroy(
    JNIEnv*, jclass, jlong handle) {
    if (handle == 0) return;
    eatom_kernel_destroy(reinterpret_cast<eatom_kernel_t*>(handle));
}

JNIEXPORT jint JNICALL
Java_com_easyhelpcare_easyatom_EasyAtomNative_nativeIngest(
    JNIEnv* env, jclass, jlong handle, jstring name) {
    if (handle == 0) return EATOM_ERR_NULL;
    JStringUTF n(env, name);
    return eatom_kernel_ingest(reinterpret_cast<eatom_kernel_t*>(handle),
                               n.c_str());
}

JNIEXPORT jint JNICALL
Java_com_easyhelpcare_easyatom_EasyAtomNative_nativeQueryArgmax(
    JNIEnv* env, jclass, jlong handle,
    jobjectArray roles, jobjectArray fillers, jstring queryRole,
    jobjectArray candidates, jboolean autoingest) {
    if (handle == 0) return -1;
    JStringArray r(env, roles), f(env, fillers), c(env, candidates);
    JStringUTF qr(env, queryRole);
    if (r.size() != f.size()) return EATOM_ERR_INVALID_ARG;
    size_t winner = 0;
    int rc = eatom_kernel_query_pairs_argmax(
        reinterpret_cast<eatom_kernel_t*>(handle),
        r.data(), r.size(), f.data(),
        qr.c_str(),
        c.data(), c.size(),
        autoingest ? 1 : 0,
        &winner);
    if (rc != EATOM_OK) return rc;
    return static_cast<jint>(winner);
}

JNIEXPORT jdoubleArray JNICALL
Java_com_easyhelpcare_easyatom_EasyAtomNative_nativeQueryProbs(
    JNIEnv* env, jclass, jlong handle,
    jobjectArray roles, jobjectArray fillers, jstring queryRole,
    jobjectArray candidates, jboolean autoingest) {
    if (handle == 0) return nullptr;
    JStringArray r(env, roles), f(env, fillers), c(env, candidates);
    JStringUTF qr(env, queryRole);
    if (r.size() != f.size()) return nullptr;
    std::vector<double> probs(c.size(), 0.0);
    int rc = eatom_kernel_query_pairs_probs(
        reinterpret_cast<eatom_kernel_t*>(handle),
        r.data(), r.size(), f.data(),
        qr.c_str(),
        c.data(), c.size(),
        autoingest ? 1 : 0,
        probs.data());
    if (rc != EATOM_OK) return nullptr;
    jdoubleArray out = env->NewDoubleArray(static_cast<jsize>(probs.size()));
    if (!out) return nullptr;
    env->SetDoubleArrayRegion(out, 0, static_cast<jsize>(probs.size()),
                              probs.data());
    return out;
}

}  // extern "C"

#endif  // EASYATOM_BUILD_JNI
