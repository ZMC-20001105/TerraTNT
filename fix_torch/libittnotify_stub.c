// Stub implementation for Intel VTune iJIT API
// This provides all missing Intel ITT symbols

int iJIT_NotifyEvent(int event_type, void *EventSpecificData) {
    return 1;
}

int iJIT_IsProfilingActive(void) {
    return 0;  // Return 0 to indicate profiling is not active
}

unsigned int iJIT_GetNewMethodID(void) {
    static unsigned int method_id = 1000;
    return method_id++;
}

void __itt_pause(void) {
    // Do nothing
}

void __itt_resume(void) {
    // Do nothing
}

void* __itt_domain_create(const char *name) {
    return (void*)0;
}

void* __itt_string_handle_create(const char *name) {
    return (void*)0;
}

void __itt_task_begin(void *domain, void *id, void *parent, void *name) {
    // Do nothing
}

void __itt_task_end(void *domain) {
    // Do nothing
}
