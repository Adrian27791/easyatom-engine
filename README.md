# EasyAtom Engine — Q-Kernel Sintético

Motor de razonamiento geométrico independiente de la app EasyHelpCare.
C++20 puro, sin dependencias externas en el núcleo.

## Visión

> *No almacenamos respuestas. Almacenamos las leyes que las generan.*

El motor codifica conocimiento como **operadores matemáticos**
(álgebras de Clifford, espacios de Hilbert simulados, geometría de la
información, homología persistente, operadores de Koopman) y "razona"
mediante **composición** de esos operadores, no mediante recuperación de
texto ni predicción estadística de tokens.

## Estructura

```
easyatom-engine/
├── CMakeLists.txt              ← build raíz (host)
├── include/easyatom/           ← API pública header-only
│   └── clifford/
│       └── multivector.hpp     ← Ladrillo 0: Cl(p,q) genérico
├── src/                        ← implementaciones .cpp futuras
├── tests/
│   ├── CMakeLists.txt
│   ├── test_framework.hpp      ← framework mínimo (sin gtest)
│   ├── test_main.cpp
│   └── test_clifford.cpp       ← tests numéricos de Cl(p,q)
└── docs/                       ← notas matemáticas
```

## Roadmap de ladrillos (orden estricto)

| # | Ladrillo | Estado |
|---|---|---|
| 0 | Multivectores de Clifford `Cl(p,q)` + producto geométrico | ✅ |
| 1 | Espacio de Hilbert simulado `H_D` + estados normalizados | ✅ |
| 2 | Operadores fundamentales geométricos (bind/bundle/permute/unbind) | ✅ |
| 3 | Métrica: Fisher-Rao + α-conexiones de Amari | ✅ |
| 4 | Topología: homología persistente (Vietoris-Rips, H_0/b_1, bottleneck) | ✅ |
| 5 | Dinámica: operador de Koopman (EDMD) | ✅ |
| 6 | Compilación de leyes (SINDy / STLSQ) | ✅ |
| 7 | API pública del Q-Kernel: `ingest / compose / collapse` | ✅ |
| 8 | Bindings: C ABI (`include/easyatom/c_api.h`) + JNI Android | ✅ |

## Reglas de construcción

1. **Cada ladrillo tiene tests numéricos exactos antes del siguiente.**
2. **Sin dependencias externas en el núcleo.** Tests permitidos solo con
   framework propio mínimo.
3. **C++20 estándar puro.** Nada específico de plataforma en `include/`.
4. **No alucinaciones por construcción.** Cada operación que no tiene
   resultado matemáticamente definido devuelve un error explícito, no
   una aproximación silenciosa.

## Build (host Windows con clang++ del NDK)

```powershell
cd easyatom-engine
cmake -B build -G Ninja `
  -DCMAKE_CXX_COMPILER="C:/Users/DnR/AppData/Local/Android/SDK/ndk/29.0.14033849/toolchains/llvm/prebuilt/windows-x86_64/bin/clang++.exe" `
  -DCMAKE_C_COMPILER="C:/Users/DnR/AppData/Local/Android/SDK/ndk/29.0.14033849/toolchains/llvm/prebuilt/windows-x86_64/bin/clang.exe"
cmake --build build
ctest --test-dir build --output-on-failure
```

Si Ninja no está disponible, usar generador `"MinGW Makefiles"` o
`"Unix Makefiles"`.
