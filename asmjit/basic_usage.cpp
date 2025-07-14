#include <asmjit/asmjit.h>
#include <iostream>

int main()
{
    using namespace asmjit;
    JitRuntime rt;
    CodeHolder code;

    code.init(rt.environment());

    x86::Assembler as(&code);

    FuncDetail func;
    func.init(FuncSignatureT<int, int, int>(CallConvId::kHost), rt.environment());

    FuncFrame frame;
    frame.init(func);

    FuncArgsAssignment args(&func);
    args.assignAll(as.zdi(), as.zsi());
    args.updateFuncFrame(frame);

    frame.init(func);

    as.emitProlog(frame);
    as.emitArgsAssignment(frame, args);

    as.mov(x86::eax, x86::edi);
    as.add(x86::eax, x86::esi);
    as.ret();

    using FuncType = int(*)(int, int);
    FuncType fn;

    Error err = rt.add(&fn, &code);
    if (err) {
        std::cerr << "AsmJit error: " << DebugUtils::errorAsString(err) << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "3 + 4 = " << fn(3, 4) << std::endl;

    rt.release(fn);

    return EXIT_SUCCESS;
}
