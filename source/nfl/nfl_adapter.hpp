#pragma once
#include <memory>
#include <string>
#include <nlohmann/json.hpp>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#include <neuro-fuzzy/annbfis.h>
#include <neuro-fuzzy/annbfis_prototype.h>
#include <neuro-fuzzy/fac_prototype.h>
#include <auxiliary/roc.h>
#pragma GCC diagnostic pop
#pragma clang diagnostic pop

namespace ksi
{
	class implication;
	class t_norm;
	class dataset;
	class annbfis;
}

using json = nlohmann::json;

ksi::dataset load_json_dataset(const json &j);

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

template <typename T>
class extended_nfs : public T, public extended_nfs_base
{
public:
	using T::T;

	json train(
		const json &ds_train_json,
		int num_rules, 
		int clustering_iters, 
		int tuning_iters, 
		double learning_coeff, 
		bool normalize) override
	{
		auto ds_train = load_json_dataset(ds_train_json);
		T::_TrainDataset = ds_train;
        T::_ValidationDataset = ds_train;
		T::_train_data_file = "<streamed>";
		T::_validation_data_file = "<streamed>";
		T::_output_file = "<none>";
		T::_nRules = num_rules;
		T::_dbLearningCoefficient = learning_coeff;
		T::_bNormalisation = normalize;

		if (normalize || T::_pModyfikator)
			throw std::runtime_error{"not implemented"};

		T::createFuzzyRulebase(
			clustering_iters,
			tuning_iters,
			learning_coeff,
			T::_TrainDataset,
			T::_ValidationDataset);
		
		if (!T::_pRulebase->validate())
			throw std::runtime_error{"rulebase invalid"};

		return json{
			{"comment", "training done!"},
			{"system_name", T::get_nfs_name()},
		};
	}

	json test(const json &ds_test_json) override
	{
		auto ds_test = load_json_dataset(ds_test_json);
        T::_TestDataset = ds_test;
		T::_test_data_file = "<streamed>";

		if (!T::_pRulebase->validate())
			throw std::runtime_error{"rulebase invalid"};

		auto y_arr = json::array();
		for (std::size_t i = 0; i < ds_test.getNumberOfData(); i++)
			y_arr.push_back(T::answer(*(ds_test.getDatum(i))));

		return {
			{"y", y_arr}
		};
	}
};

class extended_annbfis : public extended_nfs<ksi::annbfis>
{
public:
	extended_annbfis(
		int nRules,
		int nClusteringIterations,
		int nTuningIterations,
		double dbLearningCoefficient,
		bool bNormalisation,
		const ksi::t_norm& tnorm,
		const ksi::implication& imp);

	extended_annbfis(
		int nRules,
		int nClusteringIterations,
		int nTuningIterations,
		double dbLearningCoefficient,
		bool bNormalisation,
		const ksi::t_norm& tnorm,
		const ksi::implication& imp,
		double positive_class,
		double negative_class,
		ksi::roc_threshold th);
};

using extended_annbfis_prototype = extended_nfs<ksi::annbfis_prototype>;

struct train_context
{
	std::unique_ptr<ksi::fac_prototype> fac_proto;
	std::unique_ptr<ksi::implication> impl;
	std::unique_ptr<ksi::t_norm> tnorm;
	std::unique_ptr<extended_nfs_base> sys;
	json train_result;
};

std::unique_ptr<train_context> make_system(
	const std::string &system_type,
	const std::string &implication,
	const std::string &tnorm,
	bool regression_task,
	int num_rules,
	int clustering_iters,
	int tuning_iters,
	bool normalize,
	double eta,
	double minkowski_coeff);
