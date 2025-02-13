#include <iostream>
#include <unordered_map>
#include <nlohmann/json.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Winconsistent-missing-override"
#include <tnorms/t-norm-product.h>
#include <implications/imp-fodor.h>
#include <implications/imp-goedel.h>
#include <implications/imp-goguen.h>
#include <implications/imp-kleene-dienes.h>
#include <implications/imp-lukasiewicz.h>
#include <implications/imp-reichenbach.h>
#include <implications/imp-rescher.h>
#include <implications/imp-zadeh.h>
#include <tnorms/t-norm.h>
#include <tnorms/t-norm-dombi.h>
#include <tnorms/t-norm-drastic.h>
#include <tnorms/t-norm-dubois-prade.h>
#include <tnorms/t-norm-einstein.h>
#include <tnorms/t-norm-fodor.h>
#include <tnorms/t-norm-frank.h>
#include <tnorms/t-norm-hamacher.h>
#include <tnorms/t-norm-lukasiewicz.h>
#include <tnorms/t-norm-min.h>
#include <tnorms/t-norm-parametrized.h>
#include <tnorms/t-norm-product.h>
#include <tnorms/t-norm-schweizer-sklar.h>
#include <tnorms/t-norm-sugeno-weber.h>
#include <tnorms/t-norm-yager.h>
#include <neuro-fuzzy/annbfis.h>
#include <common/number.h>
#include <partitions/fcm.h>
#pragma GCC diagnostic pop

using json = nlohmann::json;

ksi::dataset load_json_dataset(const json &j)
{
	ksi::dataset ds;
	int item_number = 0;

	for (const auto &datum : j)
	{
		ksi::datum d;
		for (const auto &value : datum)
			d.push_back(ksi::number{value.get<double>()});
		d.setID(item_number++);
		d.setIDincomplete(-1);
		ds.addDatum(d);	
	}

	return ds;
}

class extended_nfs_base
{
public:
	virtual ~extended_nfs_base() = default;

	virtual json train(
		const json &ds_train_json,
		int num_rules, 
		int clustering_iters, 
		int tuning_iters, 
		double learning_coeff, 
		bool normalize) = 0;

	virtual json test(const json &ds_test_json) = 0;
};

class extended_nfs : public ksi::annbfis, public extended_nfs_base
{
public:

	extended_nfs(
		int nRules,
		int nClusteringIterations,
		int nTuningIterations,
		double dbLearningCoefficient,
		bool bNormalisation,
		const ksi::t_norm& tnorm,
		const ksi::implication& imp) :
		ksi::abstract_annbfis(
			nRules,
			nClusteringIterations,
			nTuningIterations,
			dbLearningCoefficient,
			bNormalisation,
			tnorm,
			imp,
			ksi::fcm(nRules, nClusteringIterations)
		),
		ksi::annbfis(
			nRules,
			nClusteringIterations,
			nTuningIterations,
			dbLearningCoefficient,
			bNormalisation,
			tnorm,
			imp
		)
	{
	}

	json train(
		const json &ds_train_json,
		int num_rules, 
		int clustering_iters, 
		int tuning_iters, 
		double learning_coeff, 
		bool normalize) override
	{
		auto ds_train = load_json_dataset(ds_train_json);
		_TrainDataset = ds_train;
        _ValidationDataset = ds_train;
		_train_data_file = "<streamed>";
		_validation_data_file = "<streamed>";
		_output_file = "<none>";
		_nRules = num_rules;
		_dbLearningCoefficient = learning_coeff;
		_bNormalisation = normalize;

		if (normalize || _pModyfikator)
			throw std::runtime_error{"not implemented"};

		createFuzzyRulebase(
			clustering_iters,
			tuning_iters,
			learning_coeff,
			_TrainDataset,
			_ValidationDataset);
		
		if (!_pRulebase->validate())
			throw std::runtime_error{"rulebase invalid"};

		return json{
			{"comment", "training done!"},
			{"system_name", get_nfs_name()},
		};
	}

	json test(const json &ds_test_json) override
	{
		auto ds_test = load_json_dataset(ds_test_json);
        _TestDataset = ds_test;
		_test_data_file = "<streamed>";

		if (!_pRulebase->validate())
			throw std::runtime_error{"rulebase invalid"};

		auto y_arr = json::array();
		for (std::size_t i = 0; i < ds_test.getNumberOfData(); i++)
			y_arr.push_back(answer(*(ds_test.getDatum(i))));

		return {
			{"y", y_arr}
		};
	}
};

std::unique_ptr<ksi::implication> make_implication(const std::string &name)
{
	#define MATCH(val, type) do { if (name == val) return std::make_unique<type>(); } while (false)
	MATCH("fodor", ksi::imp_fodor);
	MATCH("goedel", ksi::imp_goedel);
	MATCH("goguen", ksi::imp_goguen);
	MATCH("kleene_dienes", ksi::imp_kleene_dienes);
	MATCH("lukasiewicz", ksi::imp_lukasiewicz);
	MATCH("reichenbach", ksi::imp_reichenbach);
	MATCH("rescher", ksi::imp_rescher);
	MATCH("zadeh", ksi::imp_zadeh);
	#undef MATCH
	throw std::runtime_error{"invalid implication name"};
}

std::unique_ptr<ksi::t_norm> make_tnorm(const std::string &name)
{
	#define MATCH(val, type) do { if (name == val) return std::make_unique<type>(); } while (false)
	// MATCH("dombi", ksi::t_norm_dombi);
    MATCH("drastic", ksi::t_norm_drastic);
    // MATCH("dubois_prade", ksi::t_norm_dubois_prade);
    MATCH("einstein", ksi::t_norm_einstein);
    MATCH("fodor", ksi::t_norm_fodor);
    // MATCH("frank", ksi::t_norm_frank);
    // MATCH("hamacher", ksi::t_norm_hamacher);
    MATCH("lukasiewicz", ksi::t_norm_lukasiewicz);
    MATCH("min", ksi::t_norm_min);
    // MATCH("parametrized", ksi::t_norm_parametrized);
    MATCH("product", ksi::t_norm_product);
    // MATCH("schweizer_sklar", ksi::t_norm_schweizer_sklar);
    // MATCH("sugeno_weber", ksi::t_norm_sugeno_weber);
    // MATCH("yager", ksi::t_norm_yager);
	#undef MATCH
	throw std::runtime_error{"invalid tnorm name"};
}


struct train_context
{
	std::unique_ptr<ksi::implication> impl;
	std::unique_ptr<ksi::t_norm> tnorm;
	std::unique_ptr<extended_nfs_base> sys;
	json train_result;
};

std::unique_ptr<train_context> train_system(const json &j)
{
	const auto &config = j.at("config");
	const auto &dataset = j.at("Xy_train");
	const int num_rules = config.at("num_rules");
	const int clustering_iters = config.at("clustering_iters");
	const int tuning_iters = config.at("tuning_iters");
	const bool normalize = config.at("normalize");;
	const double eta = config.at("eta");

	std::cerr << "Will train system..." << std::endl;
	auto ctx = std::make_unique<train_context>();

	ctx->impl = make_implication(config.at("implication"));
	ctx->tnorm = make_tnorm(config.at("tnorm"));
	ctx->sys = std::make_unique<extended_nfs>(
		num_rules,
		clustering_iters,
		tuning_iters,
		eta,
		normalize,
		*ctx->tnorm,
		*ctx->impl
	);

	ctx->train_result = ctx->sys->train(
		dataset,
		num_rules, 
		clustering_iters, 
		tuning_iters, 
		eta, 
		normalize
	);

	return ctx;
}

json test_system(extended_nfs_base& sys, const json &j)
{
	std::cerr << "Will test system..." << std::endl;
	return sys.test(j.at("dataset"));
}

int main()
{
	std::unique_ptr<train_context> ctx;

	while (std::cin.peek() != EOF)
	{
		if (std::cin.peek() == '\n')
		{
			std::cin.get();
			continue;
		}

		std::cerr << "Awaiting orders..." << std::endl;
		json j;
		std::cin >> j;

		if (j.at("cmd") == "train")
		{
			ctx = train_system(j);
			std::cout << ctx->train_result << std::endl;
		}
		else if (j.at("cmd") == "test")
		{
			if (!ctx || !ctx->sys)
				throw std::runtime_error{"not trained"};
			
			json test_result = test_system(*ctx->sys, j);
			std::cout << test_result << std::endl;
		}
		else
			throw std::runtime_error{"invalid cmd"};
	}

	return 0;
}
