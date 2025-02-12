#include <iostream>
#include <nlohmann/json.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Winconsistent-missing-override"
#include <tnorms/t-norm-product.h>
#include <implications/imp-reichenbach.h>
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
	// static_assert(std::is_base_of_v<ksi::neuro_fuzzy_system, T>, "Only works with NFS");
	// static_assert(std::is_base_of_v<ksi::abstract_annbfis, T>, "Only works with NFS");

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
		// auto ds_test = _TestDataset.splitDataSetVertically(_TestDataset.getNumberOfAttributes() - 1);
		// auto ds_test = _TestDataset.getNumberOfData();

		for (std::size_t i = 0; i < ds_test.getNumberOfData(); i++)
			y_arr.push_back(answer(*(ds_test.getDatum(i))));

		return {
			{"y", y_arr}
		};
	}
};


struct train_context
{

};

std::pair<std::unique_ptr<extended_nfs_base>, json> train_system(const json &j)
{
	const auto &config = j.at("config");
	const auto &dataset = j.at("Xy_train");
	const int num_rules = config.at("num_rules");
	const int clustering_iters = config.at("clustering_iters");
	const int tuning_iters = config.at("tuning_iters");
	const bool normalize = config.at("normalize");;
	const double eta = config.at("eta");

	std::cerr << "Will train system..." << std::endl;
	ksi::imp_reichenbach implication;
	ksi::t_norm_product tnorm;

	auto sys = std::unique_ptr<extended_nfs>{new extended_nfs{
		num_rules,
		clustering_iters,
		tuning_iters,
		eta,
		normalize,
		tnorm,
		implication
	}};

	auto train_result = sys->train(
		dataset,
		num_rules, 
		clustering_iters, 
		tuning_iters, 
		eta, 
		normalize
	);

	return {std::move(sys), train_result};
}

json test_system(extended_nfs_base& sys, const json &j)
{
	std::cerr << "Will test system..." << std::endl;
	return sys.test(j.at("dataset"));
}

int main()
{
	std::unique_ptr<extended_nfs_base> sys;

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
			json train_result;
			std::tie(sys, train_result) = train_system(j);
			std::cout << train_result << std::endl;
		}
		else if (j.at("cmd") == "test")
		{
			if (!sys)
				throw std::runtime_error{"not trained"};
			
			json test_result = test_system(*sys, j);
			std::cout << test_result << std::endl;
		}
		else
			throw std::runtime_error{"invalid cmd"};
	}

	return 0;
}
