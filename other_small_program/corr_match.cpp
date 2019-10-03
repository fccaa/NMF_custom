
//////////////////////////
//// Standard Headers ////
//////////////////////////

#include <iostream>
#include <ios>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <random>
#include <algorithm>

///////////////////////////////////////////////
//// Herder file for the Armadillo Library ////
///////////////////////////////////////////////

#include <armadillo>

using namespace arma;
using namespace std;

typedef vector<double>	v_d;
typedef vector<v_d>	vv_d;

////////////////////////
//// Custom Headers ////
////////////////////////

class my_ran_class
{
  ranlux48 generator;
  uniform_int_distribution<int> uniform_int;
  uniform_real_distribution<double> uniform_double;

  public:
  int set_seed(int i)
  {
    generator.seed(i);
    return 0;
  }
  int operator () (int i)
  {
    return uniform_int(generator) % i;
  }
  double operator () ()
  {
    return uniform_double(generator);
  }
  double normal(const double & mu, const double & sigma)
  {
    return normal_distribution<double>(mu, sigma)(generator);
  }
  double lognormal(const double & mu, const double & sigma)
  {
    return lognormal_distribution<double>(mu, sigma)(generator);
  }
  double exponential(const double &lambda)
  {
    return exponential_distribution<double>(lambda)(generator);
  }
};

int get_data_col(v_d &, vv_d &, string &);

int get_data_row(v_d &, vv_d &, string &);

int get_binned_data(v_d &, vv_d &, string &, const int, const int);

string get_arg(string tag, int argc, char ** argv);

int check_arg_option(string, int argc, char ** argv);

///////////////////////
//// Main Function ////
///////////////////////

int main(int argc, char * argv[])
{
  const int N_cor_len(5);

  string in_file_name, out_file_name_1, out_file_name_2;

/*

	-i to get the name of input file

  */

  if (!check_arg_option("-i", argc, argv))
  {
    cerr << "-i is missing" << endl;
    return 1;
  }

  /*

  	-os to get the name of output file for simple output

	*/

  if (!check_arg_option("-os", argc, argv))
  {
    cerr << "-os is missing" << endl;
    return 1;
  }

  /*

  	-om to get the name of output file for GNUplot map

	*/

  if (!check_arg_option("-om", argc, argv))
  {
    cerr << "-om is missing" << endl;
    return 1;
  }

  in_file_name = get_arg("-i", argc, argv);

  out_file_name_1 = get_arg("-om", argc, argv);

  out_file_name_2 = get_arg("-os", argc, argv);

  {
    ifstream fin;

    fin.open(in_file_name);

    if (!fin.good())
    {
      cerr << "Input file has problem or does not exist." << endl;
      return 1;
    }

    fin.close();

    fin.open(out_file_name_1);

    if (fin.good())
    {
      cerr << out_file_name_1 << " exists. Exit." << endl;
      return 1;
    }

    fin.close();

    fin.open(out_file_name_2);

    if (fin.good())
    {
      cerr << out_file_name_2 << " exists. Exit." << endl;
      return 1;
    }

    fin.close();
  }

  v_d data_t;

  vv_d data_signal;

  get_binned_data(data_t, data_signal, in_file_name, 4, 1);

  mat signal_matrix(data_signal.size(), data_signal[0].size());

  {
    for (int i = 0; i < signal_matrix.n_rows; ++i)
    {
      for (int j = 0; j < signal_matrix.n_cols; ++j)
      {
	signal_matrix(i,j) = data_signal[i][j];
      }
    }
  }

  v_d all_corr_time;

  vector <mat> all_corr_mat;

  for (int i = 0; (i+N_cor_len) < signal_matrix.n_rows; i += N_cor_len)
  {
    mat working_mat = signal_matrix.rows(i, i+N_cor_len-1);

    mat cov_matrix = cov(working_mat);

    mat cor_matrix(cov_matrix.n_rows, cov_matrix.n_cols);

    for (int i1 = 0; i1 < cor_matrix.n_rows; ++i1)
    {
      for (int i2 = 0; i2 < cor_matrix.n_cols; ++i2)
      {
	if (i1 == i2)
	{
	  cor_matrix(i1, i2) = 0.0;
	  continue;
	}

	if (cov_matrix(i1,i1) < 1e-10 || cov_matrix(i2,i2) < 1e-10)
	{
	  cor_matrix(i1, i2) = 0.0;
	  continue;
	}

	cor_matrix(i1,i2) = cov_matrix(i1, i2) / sqrt(fabs(cov_matrix(i1,i1)) * fabs(cov_matrix(i2,i2)));
      }
    }

    all_corr_time.push_back(data_t[i]);

    all_corr_mat.push_back(cor_matrix);

    if (data_t[i] >= 60.0) break;
  }

  mat ans_matrix(all_corr_mat.size(), all_corr_mat.size());

  for (int i = 0; i < ans_matrix.n_rows; ++i)
  {
    for (int j = 0; j < ans_matrix.n_cols; ++j)
    {
      ans_matrix(i,j) = (sum(sum((all_corr_mat[i] % all_corr_mat[j])))) / (all_corr_mat[0].n_rows - 1.0) / (all_corr_mat[0].n_cols) ;
    }
  }

  {
    ofstream fout;
    fout.open(out_file_name_1);

    const double dt = 0.5*(all_corr_time[1]-all_corr_time[0]);

    for (int i = 0; i < ans_matrix.n_rows; ++i)
    {
      for (int j = 0; j < ans_matrix.n_cols; ++j)
      {
	fout << all_corr_time[i]-dt << "\t" << all_corr_time[j]-dt << "\t" << ((i!=j)?ans_matrix(i,j):1) << endl;
	fout << all_corr_time[i]-dt << "\t" << all_corr_time[j]+dt << "\t" << ((i!=j)?ans_matrix(i,j):1) << endl;
      }
      fout << endl;

      for (int j = 0; j < ans_matrix.n_cols; ++j)
      {
	fout << all_corr_time[i]+dt << "\t" << all_corr_time[j]-dt << "\t" << ((i!=j)?ans_matrix(i,j):1) << endl;
	fout << all_corr_time[i]+dt << "\t" << all_corr_time[j]+dt << "\t" << ((i!=j)?ans_matrix(i,j):1) << endl;
      }
      fout << endl;
    }

    fout.close();
  }

  {
    ofstream fout;
    fout.open(out_file_name_2);

    for (int i = 0; i < ans_matrix.n_rows; ++i)
    {
      double ans(0);

      for (int j = 0; j < ans_matrix.n_cols; ++j)
      {
	if (i == j) continue;
	ans += ans_matrix(i,j);
      }

      fout << all_corr_time[i] << "\t" << ans << endl;
    }

    fout.close();
  }






  return EXIT_SUCCESS;
}

string get_arg(string tag, int argc, char ** argv)
{
  for (int i = 1; i < argc; ++i)
  {
    if (tag.compare(argv[i]) == 0)
    {
      if ((i+1) < argc)
      {
	return (argv[i+1]);
      }
    }
  }
  return "";
}

int check_arg_option(string tag, int argc, char ** argv)
{
  for (int i = 0; i < argc; ++i)
  {
    if (tag.compare(argv[i]) == 0)
    {
      return 1;
    }
  }
  return 0;
}

int get_data_col(v_d & data_t, vv_d & data_signal, string & data_file)
{
  data_t.clear();
  data_signal.clear();

  ifstream fin;
  fin.open(data_file);

  while (fin.good())
  {
    string str;
    getline(fin, str);
    stringstream sstr(str);

    if (fin.eof())
    {
      break;
    }

    double t_temp(0.0);
    sstr >> t_temp;

    data_t.push_back(t_temp);

    v_d temp_vector;

    while (sstr.good())
    {
      double temp(0.0);

      int ongoin(0);

      for (int i = 1; i < 200; ++i)
      {
	if (int(str.c_str()[int(sstr.tellg())+i]) > 30)
	{
	  ongoin = 1;
	  break;
	}
	if (int(str.c_str()[int(sstr.tellg())+i]) == 0) break;
      }

      if (ongoin < 1) break;

      sstr >> temp;
      temp_vector.push_back(temp);
    };

    data_signal.push_back(temp_vector);

  };

  fin.close();

  return 0;
}

int get_data_row(v_d & data_t, vv_d & data_signal, string & data_file)
{
  vv_d data_signal_tmp;
  data_signal_tmp.clear();

  ifstream fin;
  fin.open(data_file);

  while (fin.good())
  {
    string str;
    getline(fin, str);
    stringstream sstr(str);

    if (fin.eof())
    {
      break;
    }

    v_d temp_vector;

    while (sstr.good())
    {
      double temp(0.0);

      int ongoin(0);

      for (int i = 1; i < 200; ++i)
      {
	if (int(str.c_str()[int(sstr.tellg())+i]) > 30)
	{
	  ongoin = 1;
	  break;
	}
	if (int(str.c_str()[int(sstr.tellg())+i]) == 0) break;
      }

      if (ongoin < 1) break;

      sstr >> temp;
      temp_vector.push_back(temp);
    };

    data_signal_tmp.push_back(temp_vector);
  };

  fin.close();

  data_signal.clear();
  data_signal.assign(data_signal_tmp[0].size(), v_d(data_signal_tmp.size(), 0));

  for (int i = 0; i < data_signal.size(); ++i)
  {
    for (int j = 0; j < data_signal[i].size(); ++j)
    {
      data_signal[i][j] = data_signal_tmp[j][i];
    }
  }

  data_t.clear();

  for (int i = 0; i < data_signal.size(); ++i)
  {
    data_t.push_back(i*0.1);
  }

  return 0;
}

int get_binned_data(v_d & binned_t, vv_d & binned_signal, string & data_file, const int bin_width, const int col)
{
  vv_d data_signal_ori;
  v_d data_t;

  if (col)
  {
    get_data_col(data_t, data_signal_ori, data_file);
  }
  else
  {
    get_data_row(data_t, data_signal_ori, data_file);
  }

  vv_d data_signal(data_signal_ori.size(), v_d(data_signal_ori[0].size(), 0));

  {
    double sd(0);
    int count(0);

    for (int i = 0; i < data_signal_ori.size(); ++i)
    {
      for (int j = 0; j < data_signal_ori[i].size(); ++j)
      {
	sd += pow(data_signal_ori[i][j] , 2);
	count ++;
      }
    }

    sd = sqrt(sd / count);

    for (int i = 0; i < data_signal_ori.size(); ++i)
    {
      for (int j = 0; j < data_signal_ori[i].size(); ++j)
      {
	data_signal[i][j] = data_signal_ori[i][j] / sd;
      }
    }
  }

  const int N_bins = data_t.size() / bin_width;

  for (int i = 0; i < N_bins; ++i)
  {
    double temp(0.0);
    for (int j = 0; j < bin_width; ++j)
    {
      temp += data_t[i*bin_width + j];
    }
    temp /= double(bin_width);
    binned_t.push_back(temp);

    v_d temp_v(data_signal[0].size(), 0);
    for (unsigned int j = 0; j < temp_v.size(); ++j)
    {
      for (int k = 0; k < bin_width; ++k)
      {
	temp_v[j] += data_signal[i*bin_width+k][j];
      }
      temp_v[j] /= double(bin_width);
    }
    binned_signal.push_back(temp_v);
  }

  return 0;
}


