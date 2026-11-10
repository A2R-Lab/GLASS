Supported platforms
===================

GLASS states what has been run, not what should work. Correctness evidence
comes at three levels of decreasing strength, and this page lists every
device, toolkit and host the library has been checked on at each level. A
combination that is not listed has not been run. The headers carry no
architecture gate below the listed devices, so an unlisted combination is
expected to compile and run, but nothing here claims that it does.

Attested
--------

A signed receipt that CI verifies on every push: the signature against the
signer's GitHub keys, the fingerprint of the library and test sources against
the attested commit, and the exact outcome of every test. A receipt is a
keyholder's attestation that the suite passed on that tree, not proof of GPU
execution; :doc:`../tutorials/running_tests` describes the mechanics and the
trust model.

.. list-table::
   :header-rows: 1
   :widths: 20 24 30 26

   * - Device
     - Toolkit and host
     - Scope
     - Receipt
   * - RTX 5090 (``sm_120``)
     - CUDA 13.2, x86-64, MathDx 26.03
     - All eight shards, plus the ``compute_75`` PTX-JIT lane of the
       restricted-caller regression
     - ``test/gpu-proof.json``, refreshed with every change to a fingerprinted
       path
   * - Jetson AGX Orin (``sm_87``)
     - JetPack 7, CUDA 13.2, aarch64, MathDx 26.03
     - All eight shards
     - ``test/gpu-proof-sm87.json``, recorded at the commit it names
   * - Jetson AGX Xavier (``sm_72``)
     - JetPack 5.1, CUDA 11.4, aarch64, no MathDx
     - All eight shards; every cuBLASDx and cuSOLVERDx test and the
       ``compute_75`` JIT tests skip (189 of 5257), as the exact set in
       ``test/expected_skips_sm72.txt``
     - ``test/gpu-proof-sm72.json``, recorded at the commit it names

The desktop receipt is refreshed with every change to the library or the
tests. The Jetson lanes are recorded on the device without a key, signed on
the keyholder's box, and refreshed less often, so each names an earlier
commit than the desktop receipt. CI verifies a lane against the exact tree it
attests, requires that commit to be an ancestor of the commit under test, and
rejects a lane older than 180 days (``test/gpu-proof-policy-lane.json``).
Between refreshes, a lane says that the suite passed on that device at that
commit, and nothing about changes since.

Exercised
---------

Evidence attached to a dated run, without a receipt. The reload-to-plain-load
mutant of the cooperative handoff and the selected-operand regression were run
by hand on all three devices above on 2026-10-31 (:doc:`../tutorials/running_tests`).
No device beyond those three has been exercised.

Compiles
--------

GitHub-hosted CI compiles the header surface, every MathDx-free example and
the installed package with ``nvcc`` alone, for each cell below, on every
push. A cell says the code compiles for that toolkit and architecture; it says
nothing about running there.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Toolkit
     - Architectures
   * - CUDA 13.2
     - ``sm_75``, ``sm_86``, ``sm_87``, ``sm_90``, ``sm_100``, ``sm_120``
   * - CUDA 12.4
     - ``sm_72``, ``sm_75``, ``sm_86``, ``sm_89``, ``sm_90``
   * - CUDA 11.8
     - ``sm_70``, ``sm_72``, ``sm_75``, ``sm_86``, ``sm_89``

CUDA 13 cannot target ``sm_70`` or ``sm_72``, so the Xavier lane always
builds with CUDA 11.4. The pure-SIMT surface needs ``nvcc -std=c++17`` and
nothing else; the ``glass::nvidia::`` backend needs MathDx, which exists for
CUDA 12 and 13 only.

Not claimed
-----------

Windows and macOS hosts; running on Hopper (``sm_90``) or data-center
Blackwell (``sm_100``), which compile only; CUDA toolkits before 11.4; MathDx
releases before 26.03. If you run the suite on a combination that is not
listed, an issue with the device, toolkit and outcome is the way to add a row.
