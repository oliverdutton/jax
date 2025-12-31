/*
 * Minimal C extension to register a stub for tpu_custom_call on CPU.
 *
 * This provides a no-op implementation that allows compilation to succeed,
 * but the code won't actually execute correctly - it's for testing/IR inspection only.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <string.h>

/* XLA Custom Call API signature (v0 - untyped) */
void tpu_custom_call_stub(void* out, const void** in, const char* opaque, size_t opaque_len) {
    /* This is a stub - it doesn't actually do anything useful */
    fprintf(stderr, "WARNING: tpu_custom_call stub invoked - this is a no-op!\n");
    fprintf(stderr, "  Opaque data length: %zu bytes\n", opaque_len);
    fprintf(stderr, "  This code cannot execute without a real TPU backend.\n");

    /* Don't touch output - leave uninitialized */
    /* In a real implementation, this would parse opaque data and execute the kernel */
}

/* Python module initialization */
static PyMethodDef module_methods[] = {
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "tpu_stub_extension",
    "Provides a stub for tpu_custom_call to enable compilation on CPU",
    -1,
    module_methods
};

PyMODINIT_FUNC PyInit_tpu_stub_extension(void) {
    PyObject *module = PyModule_Create(&module_def);
    if (module == NULL) {
        return NULL;
    }

    /* Create a PyCapsule containing the function pointer */
    PyObject *capsule = PyCapsule_New((void*)tpu_custom_call_stub,
                                     "xla._CUSTOM_CALL_TARGET",
                                     NULL);
    if (capsule == NULL) {
        Py_DECREF(module);
        return NULL;
    }

    /* Add the capsule to the module */
    if (PyModule_AddObject(module, "tpu_custom_call_capsule", capsule) < 0) {
        Py_DECREF(capsule);
        Py_DECREF(module);
        return NULL;
    }

    return module;
}
