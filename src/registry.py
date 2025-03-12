class Component:
    def __init__(self, name, real_name):
        self.name = name
        self.real_name = real_name
        return


class SolverName:
    """ solvers """

    """ classical methods """

    @classmethod
    def simulated_annealing(cls):
        return ['sa', 'Simulated Annealing']

    @classmethod
    def bayesian_optimization(cls):
        return ['bo', 'Bayesian Optimization']

    @classmethod
    def particle_swarm_optimization(cls):
        return ['pso', 'Particle Swarm Optimization']

    """ qubo solvers """

    @classmethod
    def amplify(cls):
        return ['amp', 'Amplify']

    @classmethod
    def dwave(cls):
        return ['dw', 'DWave']

    @classmethod
    def leap_hybrid(cls):
        return ['leap_h', 'LeapHybrid']

    @classmethod
    def amplify_dwave(cls):
        return [f'{cls.amplify()[0]}_{cls.dwave()[0]}',
                f'{cls.amplify()[1]}_{cls.dwave()[1]}']

    @classmethod
    def amplify_leap_hybrid(cls):
        return [f'{cls.amplify()[0]}_{cls.leap_hybrid()[0]}',
                f'{cls.amplify()[1]}_{cls.leap_hybrid()[1]}']

    """ instantiation of classical solvers """

    @classmethod
    def hyperopt(cls):
        return ['hy', 'hyperopt']

    @classmethod
    def hyperopt_bayesian_optimization(cls):
        return [f'{SolverName.hyperopt()[0]}_{cls.bayesian_optimization()[0]}',
                f'{SolverName.hyperopt()[1]}_{cls.bayesian_optimization()[1]}']

    @classmethod
    def hyperopt_simulated_annealing(cls):
        return [f'{SolverName.hyperopt()[0]}_{cls.simulated_annealing()[0]}',
                f'{SolverName.hyperopt()[1]}_{cls.simulated_annealing()[1]}']

    @classmethod
    def hyperopt_onehot_bayesian_optimization(cls):
        return [f'{SolverName.hyperopt()[0]}4oh_{cls.bayesian_optimization()[0]}',
                f'{SolverName.hyperopt()[1]}4oh_{cls.bayesian_optimization()[1]}']

    @classmethod
    def hyperopt_onehot_simulated_annealing(cls):
        return [f'{SolverName.hyperopt()[0]}4oh_{cls.simulated_annealing()[0]}',
                f'{SolverName.hyperopt()[1]}4oh_{cls.simulated_annealing()[1]}']

    """ classical solver for infinite solution set """

    @classmethod
    def hyperopt_set_bayesian_optimization(cls):
        return [f'{SolverName.hyperopt()[0]}4set_{cls.bayesian_optimization()[0]}',
                f'{SolverName.hyperopt()[1]}4set_{cls.bayesian_optimization()[1]}']

    @classmethod
    def hyperopt_set_simulated_annealing(cls):
        return [f'{SolverName.hyperopt()[0]}4set_{cls.simulated_annealing()[0]}',
                f'{SolverName.hyperopt()[1]}4set_{cls.simulated_annealing()[1]}']


qubo_list = [SolverName.amplify()[0], SolverName.amplify_dwave()[0], SolverName.amplify_leap_hybrid()[0],
             SolverName.dwave()[0], SolverName.leap_hybrid()[0]]


class OptimizerName:
    @classmethod
    def random_search(cls):
        return ['rg', 'Random Generation']

    @classmethod
    def crystal_params_vector(cls):
        return ['cpv', 'Crystal Parameters']

    @classmethod
    def crystal_params_vector_opt(cls):
        return [f'{OptimizerName.crystal_params_vector()[0]}_fit',
                f'{OptimizerName.crystal_params_vector()[1]}_Fitting']

    @classmethod
    def cryspy_rs(cls):
        return ['cryspy_rs', 'Cryspy_RS']

    @classmethod
    def calypso(cls):
        return ['calypso', 'CALYPSO']

    @classmethod
    def onehot(cls):
        return ['oh', 'Onehot']


class RegressionName:
    @classmethod
    def sim(cls):
        return ['sim', 'Simple']

    @classmethod
    def fm(cls):
        return ['fm', 'FacMach']

    @classmethod
    def output(cls, model_order, model_name, real_model_name, k_fm):
        return [f'{model_order}-{model_name}{k_fm}',
                f'{model_order}-Model{real_model_name}K{k_fm}']

    @classmethod
    def simple_model(cls, model_order):
        return cls.output(model_order, cls.sim()[0], cls.sim()[1], None)

    @classmethod
    def factorization_machine(cls, model_order, k_fm):
        return cls.output(model_order, cls.fm()[0], cls.fm()[1],  k_fm)


class EnergyEvaluatorName:
    @classmethod
    def m3gnet(cls):
        return 'M3GNet'

    @classmethod
    def chgnet(cls):
        return 'CHGNet'

    @classmethod
    def orb(cls):
        return 'OrbFF'
