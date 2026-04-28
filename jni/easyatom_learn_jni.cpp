// =============================================================================
// EasyAtom · L39 — Bindings JNI para la capa de aprendizaje (LearnSession).
// =============================================================================
//
// Solo se compila como parte del .so de Android (CMakeLists.txt define
// EASYATOM_BUILD_JNI). Espeja eatom_learn_* a Kotlin via la clase
//   com.easyhelpcare.easyatom.EasyAtomLearnNative
//
// Convencion de retorno para metodos con varios out-params: long[] con
// posiciones documentadas. Errores -> array vacio (length 0) o codigo int.

#ifdef EASYATOM_BUILD_JNI

#include <jni.h>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "easyatom/c_api_learn.h"

namespace {

class JStringUTF {
public:
    JStringUTF(JNIEnv* env, jstring s) : env_(env), s_(s) {
        c_ = (s_ != nullptr) ? env_->GetStringUTFChars(s_, nullptr) : nullptr;
    }
    ~JStringUTF() { if (c_) env_->ReleaseStringUTFChars(s_, c_); }
    JStringUTF(const JStringUTF&) = delete;
    JStringUTF& operator=(const JStringUTF&) = delete;
    [[nodiscard]] const char* c_str() const noexcept { return c_; }

private:
    JNIEnv*     env_;
    jstring     s_;
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
    [[nodiscard]] std::size_t        size() const { return ptrs_.size(); }

private:
    JNIEnv*                                                env_;
    jsize                                                  n_ = 0;
    std::vector<std::tuple<JNIEnv*, jstring, const char*>> owned_;
    std::vector<const char*>                               ptrs_;
};

jlongArray make_long_array(JNIEnv* env, const std::vector<jlong>& v) {
    jlongArray out = env->NewLongArray(static_cast<jsize>(v.size()));
    if (!out) return nullptr;
    if (!v.empty())
        env->SetLongArrayRegion(out, 0, static_cast<jsize>(v.size()), v.data());
    return out;
}

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_easyhelpcare_easyatom_EasyAtomLearnNative_nativeCreate(
    JNIEnv*, jclass, jint dim, jlong seed) {
    if (dim <= 0) return 0;
    auto* s = eatom_learn_create(static_cast<size_t>(dim),
                                 static_cast<uint64_t>(seed));
    return reinterpret_cast<jlong>(s);
}

JNIEXPORT void JNICALL
Java_com_easyhelpcare_easyatom_EasyAtomLearnNative_nativeDestroy(
    JNIEnv*, jclass, jlong handle) {
    if (handle == 0) return;
    eatom_learn_destroy(reinterpret_cast<eatom_learn_t*>(handle));
}

JNIEXPORT jlong JNICALL
Java_com_easyhelpcare_easyatom_EasyAtomLearnNative_nativeCodebookSize(
    JNIEnv*, jclass, jlong handle) {
    if (handle == 0) return 0;
    return static_cast<jlong>(eatom_learn_codebook_size(
        reinterpret_cast<eatom_learn_t*>(handle)));
}

JNIEXPORT jlong JNICALL
Java_com_easyhelpcare_easyatom_EasyAtomLearnNative_nativePendingCount(
    JNIEnv*, jclass, jlong handle) {
    if (handle == 0) return 0;
    return static_cast<jlong>(eatom_learn_pending_count(
        reinterpret_cast<eatom_learn_t*>(handle)));
}

// nativeIngestTexts(handle, String[] texts) -> long[6]:
//   [0]=status, [1]=total, [2]=compiled, [3]=accepted, [4]=rejected, [5]=failed
JNIEXPORT jlongArray JNICALL
Java_com_easyhelpcare_easyatom_EasyAtomLearnNative_nativeIngestTexts(
    JNIEnv* env, jclass, jlong handle, jobjectArray texts) {
    if (handle == 0)
        return make_long_array(env, {EATOM_ERR_NULL, 0, 0, 0, 0, 0});
    JStringArray ts(env, texts);
    size_t total = 0, compiled = 0, accepted = 0, rejected = 0, failed = 0;
    int rc = eatom_learn_ingest_texts(
        reinterpret_cast<eatom_learn_t*>(handle),
        ts.data(), ts.size(),
        &total, &compiled, &accepted, &rejected, &failed);
    return make_long_array(env, {
        rc,
        static_cast<jlong>(total),
        static_cast<jlong>(compiled),
        static_cast<jlong>(accepted),
        static_cast<jlong>(rejected),
        static_cast<jlong>(failed),
    });
}

// nativeDetectTopicsAndEnqueue(handle, String[] topics, double theta) -> long[2]:
//   [0]=status, [1]=enqueued
JNIEXPORT jlongArray JNICALL
Java_com_easyhelpcare_easyatom_EasyAtomLearnNative_nativeDetectTopicsAndEnqueue(
    JNIEnv* env, jclass, jlong handle, jobjectArray topics, jdouble theta) {
    if (handle == 0) return make_long_array(env, {EATOM_ERR_NULL, 0});
    JStringArray ts(env, topics);
    size_t enq = 0;
    int rc = eatom_learn_detect_topics_and_enqueue(
        reinterpret_cast<eatom_learn_t*>(handle),
        ts.data(), ts.size(), static_cast<double>(theta), &enq);
    return make_long_array(env, {rc, static_cast<jlong>(enq)});
}

// nativeRunAutoLoop(handle, String[] probes, double gapTheta, long kTop, long maxIters)
//   -> long[6]: [0]=status, [1]=iters, [2]=gapsDetected, [3]=proposals,
//               [4]=accepted, [5]=rejectedCoherence
JNIEXPORT jlongArray JNICALL
Java_com_easyhelpcare_easyatom_EasyAtomLearnNative_nativeRunAutoLoop(
    JNIEnv* env, jclass, jlong handle, jobjectArray probes,
    jdouble gapTheta, jlong kTop, jlong maxIters) {
    if (handle == 0)
        return make_long_array(env, {EATOM_ERR_NULL, 0, 0, 0, 0, 0});
    JStringArray ps(env, probes);
    size_t it = 0, gd = 0, prop = 0, acc = 0, rej = 0;
    int rc = eatom_learn_run_autoloop(
        reinterpret_cast<eatom_learn_t*>(handle),
        ps.data(), ps.size(),
        static_cast<double>(gapTheta),
        static_cast<size_t>(kTop > 0 ? kTop : 0),
        static_cast<size_t>(maxIters > 0 ? maxIters : 0),
        &it, &gd, &prop, &acc, &rej);
    return make_long_array(env, {
        rc,
        static_cast<jlong>(it),
        static_cast<jlong>(gd),
        static_cast<jlong>(prop),
        static_cast<jlong>(acc),
        static_cast<jlong>(rej),
    });
}

// nativeRunAutoLoopRanked(handle, probes, gapTheta, kMin, kMax, maxIters, mode)
//   -> long[7]: [0]=status, [1]=iters, [2]=gapsDetected,
//               [3]=candidatesTotal, [4]=candidatesUnique,
//               [5]=accepted, [6]=rejectedCoherence
JNIEXPORT jlongArray JNICALL
Java_com_easyhelpcare_easyatom_EasyAtomLearnNative_nativeRunAutoLoopRanked(
    JNIEnv* env, jclass, jlong handle, jobjectArray probes,
    jdouble gapTheta, jlong kMin, jlong kMax, jlong maxIters, jint mode) {
    if (handle == 0)
        return make_long_array(env, {EATOM_ERR_NULL, 0, 0, 0, 0, 0, 0});
    JStringArray ps(env, probes);
    size_t it = 0, gd = 0, ct = 0, cu = 0, acc = 0, rej = 0;
    int rc = eatom_learn_run_autoloop_ranked(
        reinterpret_cast<eatom_learn_t*>(handle),
        ps.data(), ps.size(),
        static_cast<double>(gapTheta),
        static_cast<size_t>(kMin > 0 ? kMin : 0),
        static_cast<size_t>(kMax > 0 ? kMax : 0),
        static_cast<size_t>(maxIters > 0 ? maxIters : 0),
        static_cast<int>(mode),
        &it, &gd, &ct, &cu, &acc, &rej);
    return make_long_array(env, {
        rc,
        static_cast<jlong>(it),
        static_cast<jlong>(gd),
        static_cast<jlong>(ct),
        static_cast<jlong>(cu),
        static_cast<jlong>(acc),
        static_cast<jlong>(rej),
    });
}

}  // extern "C"

#endif  // EASYATOM_BUILD_JNI
