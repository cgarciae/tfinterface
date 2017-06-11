FROM python27

RUN pip install numpy tensorflow git+https://github.com/cgarciae/tfinterface.git@develop
