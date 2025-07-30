import os
import re
import shutil
import subprocess
import time
import warnings

import numpy as np
from rpy2.robjects.packages import importr
from rpy2 import robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter
import rpy2.situation

__all__ = ['get_R_attr_from_NamedList', 'get_R_attr_from_ListVector', 'check_R_TMB', 'model']


def get_R_attr_from_NamedList(obj, attr):
    '''
    Convenience function to return a named attribute from an R object (ListVector)
    e.g. get_R_attr(myModel.TMB.model, 'hessian') would return the equivalent of model$hessian
    '''
    idx = list(obj.names()).index(attr)
    return obj[idx]

def get_R_attr_from_ListVector(obj, attr):
    idx = obj.names.index(attr)
    return obj[idx]

def check_R_TMB():
    '''
    Check whether R and TMB installations and paths are available
    '''

    # Start with R, but this is sort of circular, since we required rpy2
    if rpy2.situation.get_r_home() is None:
        raise Exception("R installation not found")

    # Check for R headers
    if (not 'R.h' in os.listdir(rpy2.situation.get_r_home() + "/include")):
        raise Exception("R.h not found")

    # Find TMB package
    if ro.r('find.package("TMB")') is None:
        raise Exception("TMB package doesn't exist in this R installation")

    # Find necessary TMB headers
    if not 'TMB.hpp' in os.listdir(ro.r('paste0(find.package("TMB"), "/include")')[0]):
        raise Exception("TMB headers not found in include directory. " +
                        "Possible bad installation?")

    # Finally, check for the R SO/dynlib:
    if (not 'libR.dylib' in os.listdir(rpy2.situation.get_r_home() + "/lib")) & (not 'libR.so' in os.listdir(rpy2.situation.get_r_home() + "/lib")):
        raise Exception("R shared libraries not found")

    # If all checks pass, then print the paths and exit
    print("R path: " + rpy2.situation.get_r_home())
    print("R include path: " + rpy2.situation.get_r_home() + "/include")
    print("TMB path: " + ro.r('find.package("TMB")')[0])
    print("TMB headers path: " +
          ro.r('paste0(find.package("TMB"), "/include")')[0])
    print("All necessary objects found")

    return(0)


class model:
    def __init__(self, name=None, filepath=None, codestr=None, **kwargs):
        '''
        Create a new TMB model, which utilizes an embedded R instance
        Optionally compile and load model upon instantiation if passing in filepath or codestr
        Parameters
        ----------
        name : str, default "TMB_{random.randint(1e10,9e10)}"
            Used to create model objects in R
        filepath : str (optional)
            Given a path to an existing .cpp file, the model will go ahead and compile it and load into R
        codestr : str (optional)
            A string containing .cpp code to be saved, compiled, and loaded into R
        **kwargs : optional
            Additional arguments passed to TMB_Model.compile()
        '''

        # make sure no hyphens in the model name, as that'll cause errors later
        if name:
            if name.find('-') != -1:
                raise Exception('"name" cannot contain hyphens.')

        # set model name
        self.name = name if name else 'TMB_{}'.format(
            np.random.randint(1e10, 9e10))

        # initiate R session
        self.R = ro

        # create TMB link
        self.TMB = importr('TMB')
        importr('Matrix')

        # create lists of data and initial parameter values for this model
        self.data = {}
        self.init = {}

        # compile if code passed
        if filepath or codestr:
            self.compile(filepath=filepath, codestr=codestr, **kwargs)

    def compile(self, filepath=None, codestr=None,
                output_dir='tmb_tmp',
                cc='g++',
                R=(rpy2.situation.get_r_home() + "/include"),
                TMB=(ro.r('paste0(find.package("TMB"), "/include")')[0]),
                LR=(rpy2.situation.get_r_home() + "/lib"),
                verbose=False,
                load=True,
                use_R_compiler=False):
        '''
        Compile TMB C++ code and load into R
        Parameters
        ----------
        filepath : str
            C++ code to compile
        codestr : str
            C++ code to save to .cpp then compile
        output_dir : str, default 'tmb_tmp'
            output directory for .cpp and .o
        cc : str, default 'g++'
            C++ compiler to use
        R : str, default 'rin.R_HOME + "/include"'
            location of R headers as picked up by rpy2 
            Note: R must be built with shared libraries
                  See http://stackoverflow.com/a/13224980/1028347
        TMB : str, default 'ro.r('paste0(find.package("TMB"), "/include")')[0]'
            location of TMB library
        LR : str, default 'rin.R_HOME + "/lib"'
            location of R's library files
        verbose : boolean, default False
            print compiler warnings
        load : boolean, default True
            load the model into Python after compilation
        use_R_compiler: boolean, default False
            compile the TMB model from an R subprocess
        '''
        # time compilation
        start = time.time()

        # check arguments
        if not filepath and not codestr:
            raise Exception('No filepath or codestr found.')

        # make the output directory if it doesn't already exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # if given just a filepath, copy the code into the output directory
        if filepath:
            self.filepath = os.path.join(
                output_dir, '{name}.cpp'.format(name=self.name))
            shutil.copy(filepath, self.filepath)
            if codestr:
                warnings.warn(
                    'Both filepath and codestr specified. Ignoring codestr.')

        # otherwise write code to file
        elif codestr:
            self.filepath = '{output_dir}/{name}.cpp'.format(
                output_dir=output_dir, name=self.name)

            # only rewrite cpp if identical code found
            if os.path.isfile(self.filepath) == False or \
                    open(self.filepath, 'r').read() != codestr:
                print('Saving model to {}.'.format(self.filepath))
                with open(self.filepath, 'w') as f:
                    f.write(codestr)
            else:
                print('Using {}.'.format(self.filepath))

        # compile the model
        # NOTE: cannot just call TMB.compile unfortunately -
        # something about shared libraries
        # not being hooked up correctly inside of embedded R sessions
        # TODO: skip recompiling when model has not changed

        # compile cpp
        # If using manual build
        if not use_R_compiler:
            comp = '{cc} {include} {options} {f} -o {o}'.format(
                cc=cc,
                include='-I{R} -I{TMB}'.format(R=R, TMB=TMB),
                options='-DNDEBUG -DTMB_SAFEBOUNDS -DLIB_UNLOAD=' +
                'R_unload_{} -fpic -O3 -pipe -g -c'.format(
                    self.name),
                f='{output_dir}/{name}.cpp'.format(
                    output_dir=output_dir, name=self.name),
                o='{output_dir}/{name}.o'.format(
                    output_dir=output_dir, name=self.name))
            try:
                cmnd_output = subprocess.check_output(
                    comp, stderr=subprocess.STDOUT, shell=True)
            except subprocess.CalledProcessError as exc:
                print(comp)
                print(exc.output)
                raise Exception(
                    'Your TMB code could not compile. See error above.')
            if verbose:
                print(comp)
                print(cmnd_output)
            # create shared object
            link = '{cc} {options} -o {so} {o} {link}'.format(
                cc=cc,
                options='-shared',
                so='{output_dir}/{name}.so'.format(
                    output_dir=output_dir, name=self.name),
                o='{output_dir}/{name}.o'.format(
                    output_dir=output_dir, name=self.name),
                link='-L{LR} -lR'.format(LR=LR))
            try:
                cmnd_output = subprocess.check_output(
                    link, stderr=subprocess.STDOUT, shell=True)
            except subprocess.CalledProcessError as exc:
                print(link)
                print(exc.output)
                raise Exception(
                    'Your TMB code could not be linked. See error above.')
        elif use_R_compiler:
            tmb_compile_line = 'TMB::compile("' + self.filepath + '")'
            subprocess.run(['R', '-e', tmb_compile_line])

        # if a module of the same name has already been loaded,
        # must unload R entirely it seems
        """
        TODO: fix this so R doesn't have to be restarted, potentially 
        losing things the user has already loaded into R
        judging by https://github.com/kaskr/adcomp/issues/27 this should work:
        self.R.r('try(dyn.unload("{output_dir}/{name}.so"), silent=TRUE)'.format(
        output_dir=output_dir, name=self.name))
        but it doesn't - gives odd vector errors when trying to optimize
        """
        if self.name in [str(get_R_attr_from_ListVector(i, 'name')[0])
                         for i in self.R.r('getLoadedDLLs()')]:
            warnings.warn('A model has already been loaded into TMB.')
            warnings.warn(
                'Restarting R and reloading model to prevent conflicts.')
            self.R.r('sink("/dev/null")')
            self.R.r('try(dyn.unload("{output_dir}/{name}.so"), silent=TRUE)'.format(
                output_dir=output_dir, name=self.name))
            self.R.r('sink()')
            del self.R
            self.R = ro
            del self.TMB
            self.TMB = importr('TMB')

        # load the model into R
        if load:
            self.load_model(
                so_file='{output_dir}/{name}.so'.format(output_dir=output_dir, name=self.name))

        # output time
        print('Compiled in {:.1f}s.\n'.format(time.time()-start))

    def load_model(self, so_file=''):
        if so_file == '':
            so_file = 'tmb_tmp/{name}.so'.format(name=self.name)
        if not hasattr(self, 'filepath'):
            # assume that the cpp file is in the same directory with the same name if it wasn't specified
            self.filepath = so_file.replace('.so', '.cpp')
        self.R.r('sink("/dev/null"); library(TMB)')
        self.R.r('dyn.load("{so_file}")'.format(so_file=so_file))
        self.R.r('sink()')
        self.model_loaded = True
        self.dll = os.path.splitext(os.path.basename(so_file))[0]

    def check_inputs(self, thing):
        missing = []
        with open(self.filepath) as f:
            for l in f:
                if re.match('^PARAMETER' if thing == 'init' else '^{}'.format(thing.upper()), l.strip()):
                    i = re.search(r"\(([A-Za-z0-9_]+)\)", l.strip()).group(1)
                    if i not in getattr(self, thing).keys():
                        missing.append(i)
        if missing:
            missing.sort()
            raise Exception('''Missing the following {thing}: {missing}\n
                Assign via e.g. myModel.{thing}["a"] = np.array(1., 2., 3.)'''.format(thing=thing, missing=missing))

    def build_objective_function(self, random=[], hessian=True, **kwargs):
        '''
        Builds the model objective function
        Parameters
        ----------
        random : list, default []
            which parameters should be treated as random effects (and thus integrated out of the likelihood function)
            can also be added manually via e.g. myModel.random = ['a','b']
        hessian : boolean, default True
            whether to calculate Hessian at optimum
        **kwargs : additional arguments to be passed to MakeADFun
        '''
        # first check to make sure everything necessary has been loaded
        if not hasattr(self, 'model_loaded'):
            raise Exception(
                'Model not yet compiled/loaded. See TMB_model.compile().')
        self.check_inputs('data')
        self.check_inputs('init')

        # reload the model if it's already been built
        if hasattr(self, 'obj_fun_built'):
            try:
                del self.TMB.model
                self.R.r('dyn.load("{filepath}")'.format(
                    filepath=self.filepath.replace('.cpp', '.so')))
            except:
                pass

        # save the names of random effects
        if random or not hasattr(self, 'random'):
            random.sort()
            self.random = random

        # convert random effects to the appropriate format
        if self.random:
            kwargs['random'] = self.R.StrVector(self.random)

        # store a list of fixed effects (any parameter that is not random)
        self.fixed = [v for v in self.init.keys() if v not in self.random]

        # build the objective function
        with localconverter(ro.default_converter + numpy2ri.converter):
            self.TMB.model = self.TMB.MakeADFun(data=self.R.ListVector(self.data),
                                                parameters=self.R.ListVector(self.init), hessian=hessian,
                                                DLL=self.dll, **kwargs)

        # set obj_fun_built
        self.obj_fun_built = True

    def optimize(self, opt_fun='nlminb', method='L-BFGS-B', draws=100, verbose=False,
                 random=None, quiet=False, params=[], noparams=False, constrain=False, warning=True, **kwargs):
        '''
        Optimize the model and store results in TMB_Model.TMB.fit
        Parameters
        ----------
        opt_fun : str, default 'nlminb'
            the R optimization function to use (e.g. 'nlminb' or 'optim')
        method : str, default 'L-BGFS-B'
            method to use for optimization
        draws : int or Boolean, default 100
            if Truthy, will automatically simulate draws from the posterior
        verbose : boolean, default False
            whether to print detailed optimization state
        random : list, default []
            passed to PyMB.build_objective_function
            which parameters should be treated as random effects (and thus integrated out of the likelihood function)
            can also be added manually via e.g. myModel.random = ['a','b']
        params : list of strings, default []
            which parameters to simulate, defaults to [] which means all parameters
            list parameters by name to extract their posteriors from the model
        noparams : boolean, default False
            if True, will skip finding the means of the parameters entirely
        constrain : float or boolean, default False
            if float, will constrain any draws of a parameter to be within that many
                standard deviations of the median
        warning : bool
            print warning when there is non convergence with a model
        **kwargs : additional arguments to be passed to the R optimization function
        '''
        # time function execution
        start = time.time()

        # rebuild optimization function if new random parameters are given
        rebuild = False
        if random is not None:
            if not hasattr(self, 'random') or random != self.random:
                self.random = random
                rebuild = True

        # check to make sure the optimization function has been built
        if not hasattr(self.TMB, 'model') or rebuild:
            self.build_objective_function(random=self.random)

        # turn off warnings if verbose is not on
        with localconverter(ro.default_converter + numpy2ri.converter):
            if not verbose:
                self.R.r('''
                    function(model) {
                        model$env$silent <- TRUE
                        model$env$tracemgc <- FALSE
                        model$env$inner.control$trace <- FALSE
                    }
                ''')(self.TMB.model)

            # fit the model
            if quiet:
                self.R.r('sink("/dev/null")')
            self.TMB.fit = self.R.r[opt_fun](start=get_R_attr_from_NamedList(self.TMB.model, 'par'),
                                            objective=get_R_attr_from_NamedList(
                                                self.TMB.model, 'fn'),
                                            gradient=get_R_attr_from_NamedList(
                                                self.TMB.model, 'gr'),
                                            method=method, **kwargs)
            if quiet:
                self.R.r('sink()')
            else:
                print('\nModel optimization complete in {:.1f}s.\n'.format(
                    time.time()-start))

        # check for convergence
        self.convergence = get_R_attr_from_NamedList(
            self.TMB.fit, 'convergence'
        )[0]

        if warning and self.convergence != 0:
            print(
                '\nThe model did not successfully converge, exited with the following warning message:')
            print(self.TMB.fit[self.TMB.fit.names.index('message')][0] + '\n')

    def get_report(self, par_fixed = None):
        with localconverter(ro.default_converter + numpy2ri.converter):
            if par_fixed:
                report = self.TMB.sdreport(self.TMB.model, par_fixed)
            else:   # Uses the best
                report = self.TMB.sdreport(self.TMB.model)
        report = dict(zip(report.names(), report))
        return report

    def get_random_report(self):
        with localconverter(ro.default_converter + numpy2ri.converter):
            report = self.TMB.sdreport(self.TMB.model)
            report_with_std = self.TMB.summary_sdreport(report, select = np.array(["random"]))
        return report_with_std
    
    def get_cov_fixed(self, par_fixed):
        with localconverter(ro.default_converter + numpy2ri.converter):
            report = self.TMB.sdreport(self.TMB.model, par_fixed)
        cov_fixed = np.array(get_R_attr_from_NamedList(report, 'cov'))
        return cov_fixed

    def get_hessian(self, par_fixed):
        with localconverter(ro.default_converter + numpy2ri.converter):
            hessian_np = self.R.r['optimHess'](
                par_fixed,
                get_R_attr_from_NamedList(self.TMB.model, 'fn'),
                get_R_attr_from_NamedList(self.TMB.model, 'gr')
            )
        return hessian_np
