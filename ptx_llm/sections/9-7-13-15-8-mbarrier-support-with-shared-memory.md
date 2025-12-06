##### 9.7.13.15.8. mbarrier support with shared memory 

The following table summarizes the support of various mbarrier operations on *mbarrier objects*
located at different shared memory locations:

| mbarrier operations | `.shared::cta` | `.shared::cluster` |
| --- | --- | --- |
| `mbarrier.arrive` | Supported | Supported, cannot return result |
| `mbarrier.expect_tx` | Supported | Supported |
| `mbarrier.complete_tx` | Supported | Supported |
| Other mbarrier operations | Supported | Not supported |