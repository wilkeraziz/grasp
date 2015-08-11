# Sampling directly from the grammar


```bash
python grammarator.py 2 > g2
python -m grasp.cfg.sampler g2 out --log --start A --samples 1000 -v --experiment g2 -p -vv
python -m grasp.cfg.sampler g2 out --log --start A --samples 1000 -v --experiment g2 -p -vv --local
```
