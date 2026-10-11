#include <nanobind/nanobind.h>
#include "xla/ffi/api/c_api.h"

namespace nb = nanobind;

extern "C" XLA_FFI_Error* StateScan(XLA_FFI_CallFrame* call_frame);
extern "C" XLA_FFI_Error* ReduceYOff(XLA_FFI_CallFrame* call_frame);

template <typename T>
nb::capsule EncapsulateFfiCall(T* fn) {
    return nb::capsule(reinterpret_cast<void*>(fn), "xla._CUSTOM_CALL_TARGET");
}

NB_MODULE(state_scan_ffi, m) {
    m.doc() = "K3 v3 State Scan (N-split) + Y_off reduction for Mamba3 SISO";
    m.def("registrations", []() {
        nb::dict d;
        d["siso_state_scan"] = EncapsulateFfiCall(StateScan);
        d["reduce_y_off"]    = EncapsulateFfiCall(ReduceYOff);
        return d;
    });
    m.attr("__version__") = "0.3.0";
}
