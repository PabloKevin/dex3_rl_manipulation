# System overview: G1 apple picker, sim2real stack

`SYSTEM_OVERVIEW_sim2real_2026_10_17.pdf` explains how `~/g1_fruits_picker/apple_picker/g1_launch.sh` and the sim2real
stack work: what runs on the laptop, the Jetson and the Unitree firmware; the control chain from the policy down to the
motor drivers; every sensor and where each policy input comes from; DDS topics; concurrency; state machines; safety
stops; and a ranked list of problems and optimisations. All diagrams are plain text.

The `.html` file is the source. To rebuild the PDF:

```bash
python3 -m venv /tmp/wp && /tmp/wp/bin/pip install weasyprint
/tmp/wp/bin/python -c "import weasyprint as w; w.HTML('SYSTEM_OVERVIEW_sim2real_2026_10_17.html').write_pdf('SYSTEM_OVERVIEW_sim2real_2026_10_17.pdf')"
```

The report describes the code as it was on 2026-10-17: v20 daemon md5 95a864ae on the robot, and the g1_dual layer in
`~/g1_fruits_picker/g1_dual`.
