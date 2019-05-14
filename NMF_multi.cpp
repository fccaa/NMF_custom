
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

double energy_func(const mat &, const mat &, const mat &);

int one_iteration_add(const mat &, mat &, mat &, double);

int one_iteration_mul(const mat &, mat &, mat &, double);

int NMF(double &, const mat &, mat &, mat &, const int, const int, const int);

string get_arg(string tag, int argc, char ** argv);

int check_arg_option(string, int argc, char ** argv);

int get_significance(const mat &data_matrix, const mat &basis_matrix, 
                     const mat &coeff_matrix, const int, v_d & pattern_sign);

double get_p_from_KS_test(vector<double> & v1, vector<double> &v2);

double get_F(const vector<double> &v1, const double x1);

int reorangize_patterns(mat &, mat &, vector<double> &);

int print_help(int, char**);

///////////////////////
//// Main Function ////
///////////////////////

int main(int argc, char * argv[])
{
  /// Check if "-h" is used

  if (print_help(argc, argv)) return 1;

  /// Get Parameters ///

  /// Whether data for each order is printed

  int verbose(0);

  if (check_arg_option("-v", argc, argv))
  {
    verbose = 1;
  }

  /// Terminate the program based on AIC ?

  int AIC_smart(0);

  if (check_arg_option("-S", argc, argv))
  {
    AIC_smart = 1;
  }

  /// Ask the user if the column is the time axis

  int time_is_col(-1);

  {
    if (check_arg_option("-tc", argc, argv) || time_is_col < 0)
    {
      time_is_col = 1;
    }

    if (check_arg_option("-tr", argc, argv) || time_is_col < 0)
    {
      time_is_col = 0;
    }

    if (time_is_col < 0)
    {
      cerr << "Please suggest whether the time axis of the input data is column (-tc) or row (-tr)." << endl;
      return 1;
    }
  }

  /// Get the range of orders being attempted

  int N_order_start(-1);
  int N_order_end(-1);

  N_order_start	= atoi(get_arg("-Ns", argc, argv).c_str());
  N_order_end 	= atoi(get_arg("-Ne", argc, argv).c_str());

  if (N_order_start < 1) N_order_start = 1;
  			// If the starting order is not valid, set it to 1.

  if (N_order_end < N_order_start) N_order_end = N_order_start;
  			// If the ending order is not valid, set it to the starting order.

  /// Number of attempts for each order

  int N_attempt(0);

  N_attempt = atoi(get_arg("-Na", argc, argv).c_str());

  if (N_attempt < 1) N_attempt = 1000;
  			// If the number is not valid or undefined, set it to 1000;
  
  /// Number of iterations for each attempt

  int N_iter(0);

  N_iter = atoi(get_arg("-Ni", argc, argv).c_str());

  if (N_iter < 1) N_iter = 1000;
  
  /// Number of iterations for each attempt

  int N_shuf(0);

  N_shuf = atoi(get_arg("-Nshu", argc, argv).c_str());

  if (N_shuf < 1) N_shuf = 100;

  /// Number of bins for down-sampling

  int N_downsample(0);

  N_downsample = atoi(get_arg("-Nd", argc, argv).c_str());

  if (N_downsample < 1) N_downsample = ((time_is_col)?4:1);
  			// If the number is not valid or undefined, set it to 4 or 1
			// (depends on the orientation of the data)

  /// Get input and output file names 

  string in_file, out_file_b, out_file_c, out_file_e, out_file_s;

  in_file = get_arg("-i", argc, argv);

  out_file_b = get_arg("-ob", argc, argv);	// Output for bases
  out_file_c = get_arg("-oc", argc, argv);	// Output for occurrence
  out_file_e = get_arg("-oe", argc, argv);	// Output for cost function
  out_file_s = get_arg("-os", argc, argv);	// Output for cost function

  int output_b(0);
  int output_c(0);
  int output_e(0);
  int output_s(0);

  if (out_file_b.compare("") != 0) output_b = 1;
  if (out_file_c.compare("") != 0) output_c = 1;
  if (out_file_e.compare("") != 0) output_e = 1;
  if (out_file_s.compare("") != 0) output_s = 1;

  cerr << endl;
  cerr << "************* Summary for the NMF analysis *************" << endl;
  cerr << endl;
  cerr << "Orders Being Attempted (-Ns and -Ne)        : " << N_order_start << " - " << N_order_end << endl;
  cerr << "Number of Attempts of each order (-Na)      : " << N_attempt << endl;
  cerr << "Number of Iterations for each attempt (-Ni) : " << N_iter << endl;
  cerr << "Number of shuffles for significance (-Nshu) : " << N_shuf << endl;
  cerr << endl;
  cerr << "Input Data File (-i)                        : " << in_file << endl;
  cerr << "Time along rows (-tr) or column (-tc)       : " << ((time_is_col)?"Column":"Row") << endl;
  cerr << "Number of bins for down-sampling (-Nd)      : " << N_downsample << endl;
  cerr << endl;
  cerr << "Output for Energy Func (-oe)                : " << ((output_e == 1)?out_file_e:"No output.") << endl;
  cerr << "Output for Basis (-ob)                      : " << ((output_b == 1)?out_file_b:"No output.") << endl;
  cerr << "Output for Coefficient (-oc)                : " << ((output_c == 1)?out_file_c:"No output.") << endl;
  cerr << "Output for Significance (-os)               : " << ((output_s == 1)?out_file_s:"No output.") << endl;
  cerr << endl;
  cerr << "Print detail on screen (-v)                 : " << ((verbose)?"Yes":"No") << endl;
  cerr << "Stop attempts if AIC > AIC(1) (-S)          : " << ((AIC_smart)?"Yes":"No") << endl;
  cerr << endl;
  cerr << "This program was compiled on " << __DATE__ << "." << endl;
  cerr << endl;
  cerr << "********************************************************" << endl;
  cerr << endl;

  /// Check conditions of files

  if (in_file.compare("") == 0)
  {
    cerr << "Error: No input file specified." << endl;
    return 1;
  }

  if ( ! ifstream(in_file) )
  {
    cerr << "Error: input file \"" << in_file << "\" does not exist." << endl;
    return 1;
  }

  if ( (ifstream(out_file_b)) && output_b)
  {
    cerr << "Error: output file \"" << out_file_b << "\" exist." << endl;
    return 1;
  }

  if ( (ifstream(out_file_c)) && output_c)
  {
    cerr << "Error: output file \"" << out_file_c << "\" exist." << endl;
    return 1;
  }

  if ( (ifstream(out_file_e)) && output_e)
  {
    cerr << "Error: output file \"" << out_file_e << "\" exist." << endl;
    return 1;
  }

  if ( (ifstream(out_file_s)) && output_s)
  {
    cerr << "Error: output file \"" << out_file_s << "\" exist." << endl;
    return 1;
  }

  /// Calculation Starts

  int N_order(0);

  v_d binned_t;
  vv_d binned_signal;

  get_binned_data(binned_t, binned_signal, in_file, N_downsample, time_is_col);

  mat data_matrix(binned_signal[0].size(), binned_signal.size());

  for (int i = 0; i < data_matrix.n_rows; ++i)
  {
    for (int j = 0; j < data_matrix.n_cols; ++j)
    {
      data_matrix(i,j) = binned_signal[j][i];
    }
  }

  if (output_e)
  {
    fstream fout;

    fout.open(out_file_e, fstream::out);

    fout << noshowpos << "# Cost function, AIC and BIC of NMF" << endl;
    fout << noshowpos << "# " << endl;
    fout << noshowpos << "# " << "************* Summary for the NMF analysis *************" << endl;
    fout << noshowpos << "# " << endl;
    fout << noshowpos << "# " << "Orders Being Attempted (-Ns and -Ne)        : " << N_order_start << " - " << N_order_end << endl;
    fout << noshowpos << "# " << "Number of Attempts of each order (-Na)      : " << N_attempt << endl;
    fout << noshowpos << "# " << "Number of Iterations for each attempt (-Ni) : " << N_iter << endl;
    fout << noshowpos << "# " << "Number of shuffles for significance (-Nshu) : " << N_shuf << endl;
    fout << noshowpos << "# " << endl;
    fout << noshowpos << "# " << "Input Data File (-i)                        : " << in_file << endl;
    fout << noshowpos << "# " << "Time along rows (-tr) or column (-tc)       : " << ((time_is_col)?"Column":"Row") << endl;
    fout << noshowpos << "# " << "Number of bins for down-sampling (-Nd)      : " << N_downsample << endl;
    fout << noshowpos << "# " << endl;
    fout << noshowpos << "# " << "Output for Energy Func (-oe)                : " << ((output_e == 1)?out_file_e:"No output.") << endl;
    fout << noshowpos << "# " << "Output for Basis (-ob)                      : " << ((output_b == 1)?out_file_b:"No output.") << endl;
    fout << noshowpos << "# " << "Output for Coefficient (-oc)                : " << ((output_c == 1)?out_file_c:"No output.") << endl;
    fout << noshowpos << "# " << "Output for Significance (-os)               : " << ((output_s == 1)?out_file_s:"No output.") << endl;
    fout << noshowpos << "# " << endl;
    fout << noshowpos << "# " << "Print detail on screen (-v)                 : " << ((verbose)?"Yes":"No") << endl;
    fout << noshowpos << "# " << "Stop attempts if AIC > AIC(1) (-S)          : " << ((AIC_smart)?"Yes":"No") << endl;
    fout << noshowpos << "# " << endl;
    fout << noshowpos << "# " << "This program was compiled on " << __DATE__ << "." << endl;
    fout << noshowpos << "# " << endl;
    fout << noshowpos << "# " << "********************************************************" << endl;
    fout << noshowpos << "# " << endl;
    fout << noshowpos << "# Tag:" << endl;
    fout << noshowpos << "#Order    \tE         \tAICs      \tBIC" << endl;

    fout.close();
  }

  if (verbose)
    cout << noshowpos << "N         \tE         \tAICs      \tBIC" << endl;

  vector<double> 	Order;
  vector<double> 	Energy;
  vector<double>	AIC_s;
  vector<double>	BIC_s;
  vector<mat> 		Basis_matrix;
  vector<mat> 		Coeff_matrix;

  for (N_order = N_order_start; N_order <= (N_order_end); N_order+=1)
  {
    mat basis_matrix;

    mat coeff_matrix;

    double best_E(-1);

    NMF(best_E, data_matrix, basis_matrix, coeff_matrix, N_order, N_attempt, N_iter);

    double RSS = energy_func(data_matrix, basis_matrix, coeff_matrix);

    int n = binned_signal.size() * binned_signal[0].size();

    int k = binned_signal.size() * N_order + N_order * binned_signal[0].size();

    double AIC = n * log( RSS / n ) / 2.0 + k + k * (k + 1.0) / (n - k - 1.0);

    double BIC = n * log( RSS / n ) + k * log(n);

    if (k >= n) break;

    if (verbose)
      cout << noshowpos << N_order << "\t" << scientific << "\t" << showpos << RSS << "\t" << AIC << "\t" << BIC << endl;

    if (output_e)
    {
      fstream fout;
      fout.open(out_file_e, fstream::out | fstream::app);
      fout << noshowpos << N_order << "\t" << scientific << showpos << RSS << "\t" << AIC << "\t" << BIC << endl;
      fout.close();
    }

    Order.push_back(N_order);
    Energy.push_back(best_E);
    AIC_s.push_back(AIC);
    BIC_s.push_back(BIC);
    Basis_matrix.push_back(basis_matrix);
    Coeff_matrix.push_back(coeff_matrix);

    if (AIC > (AIC_s[0] + 1) && AIC_smart && Order.size() > 10) break;
  }

  {
    int best_idx(0);

    for (int i = 0; i < Order.size(); ++i)
    {
      if (AIC_s[i] < AIC_s[best_idx]) best_idx = i;
    }

    mat basis_matrix = Basis_matrix[best_idx];
    mat coeff_matrix = Coeff_matrix[best_idx];

    if (output_b || output_c)
    {
      vector<double> net_mag(basis_matrix.n_cols, 0);

      for (int i = 0; i < net_mag.size(); ++i)
      {
	net_mag[i] = sum(sum(basis_matrix.col(i) * coeff_matrix.row(i)));
      }

      reorangize_patterns(basis_matrix, coeff_matrix, net_mag);

      coeff_matrix = coeff_matrix.t();
    }

    if (output_b)
    {
      fstream fout;
      fout.open(out_file_b, fstream::out);

      fout << noshowpos << "# NMF Patterns" << endl;
      fout << noshowpos << "# " << endl;
      fout << noshowpos << "# " << "************* Summary for the NMF analysis *************" << endl;
      fout << noshowpos << "# " << endl;
      fout << noshowpos << "# " << "Orders Being Attempted (-Ns and -Ne)        : " << N_order_start << " - " << N_order_end << endl;
      fout << noshowpos << "# " << "Number of Attempts of each order (-Na)      : " << N_attempt << endl;
      fout << noshowpos << "# " << "Number of Iterations for each attempt (-Ni) : " << N_iter << endl;
      fout << noshowpos << "# " << "Number of shuffles for significance (-Nshu) : " << N_shuf << endl;
      fout << noshowpos << "# " << endl;
      fout << noshowpos << "# " << "Input Data File (-i)                        : " << in_file << endl;
      fout << noshowpos << "# " << "Time along rows (-tr) or column (-tc)       : " << ((time_is_col)?"Column":"Row") << endl;
      fout << noshowpos << "# " << "Number of bins for down-sampling (-Nd)      : " << N_downsample << endl;
      fout << noshowpos << "# " << endl;
      fout << noshowpos << "# " << "Output for Energy Func (-oe)                : " << ((output_e == 1)?out_file_e:"No output.") << endl;
      fout << noshowpos << "# " << "Output for Basis (-ob)                      : " << ((output_b == 1)?out_file_b:"No output.") << endl;
      fout << noshowpos << "# " << "Output for Coefficient (-oc)                : " << ((output_c == 1)?out_file_c:"No output.") << endl;
      fout << noshowpos << "# " << "Output for Significance (-os)               : " << ((output_s == 1)?out_file_s:"No output.") << endl;
      fout << noshowpos << "# " << endl;
      fout << noshowpos << "# " << "Print detail on screen (-v)                 : " << ((verbose)?"Yes":"No") << endl;
      fout << noshowpos << "# " << "Stop attempts if AIC > AIC(1) (-S)          : " << ((AIC_smart)?"Yes":"No") << endl;
      fout << noshowpos << "# " << endl;
      fout << noshowpos << "# " << "This program was compiled on " << __DATE__ << "." << endl;
      fout << noshowpos << "# " << endl;
      fout << noshowpos << "# " << "********************************************************" << endl;
      fout << noshowpos << "# " << endl;
      fout << noshowpos << "# Tag:" << endl;
      fout << noshowpos << "# Neuron  \t[Pattern #1]\t[Pattern #2] ...." << endl;

      for (int i = 0; i < basis_matrix.n_rows; ++i)
      {
	fout << noshowpos << i+1;
	for (int j = 0; j < basis_matrix.n_cols; ++j)
	{
	  fout << "\t" << scientific  << showpos << basis_matrix(i,j);
	}
	fout << noshowpos << endl;
      }
      fout.close();
    }

    if (output_c)
    {
      fstream fout;
      fout.open(out_file_c, fstream::out);

      fout << noshowpos << "# NMF Time-trace intensity" << endl;
      fout << noshowpos << "# " << endl;
      fout << noshowpos << "# " << "************* Summary for the NMF analysis *************" << endl;
      fout << noshowpos << "# " << endl;
      fout << noshowpos << "# " << "Orders Being Attempted (-Ns and -Ne)        : " << N_order_start << " - " << N_order_end << endl;
      fout << noshowpos << "# " << "Number of Attempts of each order (-Na)      : " << N_attempt << endl;
      fout << noshowpos << "# " << "Number of Iterations for each attempt (-Ni) : " << N_iter << endl;
      fout << noshowpos << "# " << "Number of shuffles for significance (-Nshu) : " << N_shuf << endl;
      fout << noshowpos << "# " << endl;
      fout << noshowpos << "# " << "Input Data File (-i)                        : " << in_file << endl;
      fout << noshowpos << "# " << "Time along rows (-tr) or column (-tc)       : " << ((time_is_col)?"Column":"Row") << endl;
      fout << noshowpos << "# " << "Number of bins for down-sampling (-Nd)      : " << N_downsample << endl;
      fout << noshowpos << "# " << endl;
      fout << noshowpos << "# " << "Output for Energy Func (-oe)                : " << ((output_e == 1)?out_file_e:"No output.") << endl;
      fout << noshowpos << "# " << "Output for Basis (-ob)                      : " << ((output_b == 1)?out_file_b:"No output.") << endl;
      fout << noshowpos << "# " << "Output for Coefficient (-oc)                : " << ((output_c == 1)?out_file_c:"No output.") << endl;
      fout << noshowpos << "# " << "Output for Significance (-os)               : " << ((output_s == 1)?out_file_s:"No output.") << endl;
      fout << noshowpos << "# " << endl;
      fout << noshowpos << "# " << "Print detail on screen (-v)                 : " << ((verbose)?"Yes":"No") << endl;
      fout << noshowpos << "# " << "Stop attempts if AIC > AIC(1) (-S)          : " << ((AIC_smart)?"Yes":"No") << endl;
      fout << noshowpos << "# " << endl;
      fout << noshowpos << "# " << "This program was compiled on " << __DATE__ << "." << endl;
      fout << noshowpos << "# " << endl;
      fout << noshowpos << "# " << "********************************************************" << endl;
      fout << noshowpos << "# " << endl;
      fout << noshowpos << "# Tag:" << endl;
      fout << noshowpos << "# Time     \t[intensity #1]\t[intensity #2] ...." << endl;

      for (int i = 0; i < coeff_matrix.n_rows; ++i)
      {
	fout << noshowpos << binned_t[i];
	for (int j = 0; j < coeff_matrix.n_cols; ++j)
	{
	  fout << "\t" << scientific << showpos << coeff_matrix(i,j);
	}
	fout << noshowpos << endl;
      }
      fout.close();
    }

    if (output_s)
    {
      vector<double> pattern_sign;
      get_significance(data_matrix, basis_matrix, coeff_matrix, 1000, pattern_sign);

      fstream fout;
      fout.open(out_file_s, fstream::out);

      fout << noshowpos << "# NMF Significance relative to distribution of overlaps " << endl;
      fout << noshowpos << "# with random shuffled data set." << endl;
      fout << noshowpos << "# " << endl;
      fout << noshowpos << "# " << "************* Summary for the NMF analysis *************" << endl;
      fout << noshowpos << "# " << endl;
      fout << noshowpos << "# " << "Orders Being Attempted (-Ns and -Ne)        : " << N_order_start << " - " << N_order_end << endl;
      fout << noshowpos << "# " << "Number of Attempts of each order (-Na)      : " << N_attempt << endl;
      fout << noshowpos << "# " << "Number of Iterations for each attempt (-Ni) : " << N_iter << endl;
      fout << noshowpos << "# " << "Number of shuffles for significance (-Nshu) : " << N_shuf << endl;
      fout << noshowpos << "# " << endl;
      fout << noshowpos << "# " << "Input Data File (-i)                        : " << in_file << endl;
      fout << noshowpos << "# " << "Time along rows (-tr) or column (-tc)       : " << ((time_is_col)?"Column":"Row") << endl;
      fout << noshowpos << "# " << "Number of bins for down-sampling (-Nd)      : " << N_downsample << endl;
      fout << noshowpos << "# " << endl;
      fout << noshowpos << "# " << "Output for Energy Func (-oe)                : " << ((output_e == 1)?out_file_e:"No output.") << endl;
      fout << noshowpos << "# " << "Output for Basis (-ob)                      : " << ((output_b == 1)?out_file_b:"No output.") << endl;
      fout << noshowpos << "# " << "Output for Coefficient (-oc)                : " << ((output_c == 1)?out_file_c:"No output.") << endl;
      fout << noshowpos << "# " << "Output for Significance (-os)               : " << ((output_s == 1)?out_file_s:"No output.") << endl;
      fout << noshowpos << "# " << endl;
      fout << noshowpos << "# " << "Print detail on screen (-v)                 : " << ((verbose)?"Yes":"No") << endl;
      fout << noshowpos << "# " << "Stop attempts if AIC > AIC(1) (-S)          : " << ((AIC_smart)?"Yes":"No") << endl;
      fout << noshowpos << "# " << endl;
      fout << noshowpos << "# " << "This program was compiled on " << __DATE__ << "." << endl;
      fout << noshowpos << "# " << endl;
      fout << noshowpos << "# " << "********************************************************" << endl;
      fout << noshowpos << "# " << endl;
      fout << noshowpos << "# Tag:" << endl;
      fout << noshowpos << "# Pattern ID     \t(1-p) Significane" << endl;
      
      for (int i = 0; i < pattern_sign.size(); ++i)
      {
	fout << noshowpos << i+1 << "\t" << scientific << showpos << pattern_sign[i] << endl;
      }

      fout.close();
    }

    cerr << "The Order with lowest AICs between "<< N_order_start << " and " << N_order_end;
    cerr << ": " << Order[best_idx] << "." << endl;

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

int one_iteration_add(const mat & data_matrix, 
                  mat & basis_matrix, mat & coeff_matrix, double rate)
{
  mat d_basis_matrix_n = data_matrix * coeff_matrix.t() ;
  mat d_basis_matrix_d = basis_matrix * coeff_matrix * coeff_matrix.t();

  mat d_coeff_matrix_n = basis_matrix.t() * data_matrix ;
  mat d_coeff_matrix_d = basis_matrix.t() * basis_matrix * coeff_matrix;

#pragma omp parallel for
  for (int i = 0; i < basis_matrix.n_rows; ++i)
  {
    for (int j = 0; j < basis_matrix.n_cols; ++j)
    {
      double temp = basis_matrix(i,j) + rate * tanh(d_basis_matrix_n(i,j) - d_basis_matrix_d(i,j));
      if (temp < 0) basis_matrix(i,j) *= d_basis_matrix_n(i,j) / ( d_basis_matrix_d(i,j) + 1e-10 );
      else basis_matrix(i,j) = temp;
    }
  }

#pragma omp parallel for
  for (int i = 0; i < coeff_matrix.n_rows; ++i)
  {
    for (int j = 0; j < coeff_matrix.n_cols; ++j)
    {
      double temp = coeff_matrix(i,j) + rate * tanh(d_coeff_matrix_n(i,j) - d_coeff_matrix_d(i,j));
      if (temp < 0) coeff_matrix(i,j) *= d_coeff_matrix_n(i,j) / ( d_coeff_matrix_d(i,j) + 1e-10 );
      else coeff_matrix(i,j) = temp;
    }
  }

  return 0;
}

int one_iteration_mul(const mat & data_matrix, 
                  mat & basis_matrix, mat & coeff_matrix, double rate)
{
  mat d_basis_matrix_n = data_matrix * coeff_matrix.t() ;
  mat d_basis_matrix_d = basis_matrix * coeff_matrix * coeff_matrix.t();

  mat d_coeff_matrix_n = basis_matrix.t() * data_matrix ;
  mat d_coeff_matrix_d = basis_matrix.t() * basis_matrix * coeff_matrix;

#pragma omp parallel for
  for (int i = 0; i < d_basis_matrix_n.n_rows; ++i)
  {
    for (int j = 0; j < d_basis_matrix_n.n_cols; ++j)
    {
      d_basis_matrix_n(i,j) /= d_basis_matrix_d(i,j) + 1e-10;
    }
  }

#pragma omp parallel for
  for (int i = 0; i < d_coeff_matrix_n.n_rows; ++i)
  {
    for (int j = 0; j < d_coeff_matrix_n.n_cols; ++j)
    {
      d_coeff_matrix_n(i,j) /= d_coeff_matrix_d(i,j) + 1e-10;
    }
  }

#pragma omp parallel for
  for (int i = 0; i < basis_matrix.n_rows; ++i)
  {
    for (int j = 0; j < basis_matrix.n_cols; ++j)
    {
      basis_matrix(i,j) *= pow(d_basis_matrix_n(i,j), rate);
    }
  }

#pragma omp parallel for
  for (int i = 0; i < coeff_matrix.n_rows; ++i)
  {
    for (int j = 0; j < coeff_matrix.n_cols; ++j)
    {
      coeff_matrix(i,j) *= pow(d_coeff_matrix_n(i,j), rate);
    }
  }

  return 0;
}

double energy_func(const mat &data_matrix, const mat &basis_matrix, 
                   const mat &coeff_matrix)
{
  mat temp(data_matrix.n_rows, data_matrix.n_cols);

  temp = data_matrix - basis_matrix * coeff_matrix;

  temp = square(temp);

  return accu(temp);
}

int NMF(double &best_E, const mat & data_matrix, mat & basis_matrix_best, 
    mat & coeff_matrix_best, const int N_order, const int attempt, const int N_iter)
{
  best_E = -1;

  mat data_matrix_nn = 0.5 * (data_matrix + abs(data_matrix));

#pragma omp parallel for
  for (int p = 0; p < (attempt+2); ++p)
  {
    my_ran_class my_ran;

    my_ran.set_seed(p+1);

    mat basis_matrix_tmp(data_matrix_nn.n_rows, N_order);
    mat coeff_matrix_tmp(N_order, data_matrix_nn.n_cols);

    {
      for (int i = 0; i < basis_matrix_tmp.n_rows; ++i)
      {
	for (int j = 0; j < basis_matrix_tmp.n_cols; ++j)
	{
	  basis_matrix_tmp(i,j) = 1.0 + my_ran();
	}
      }
      for (int i = 0; i < coeff_matrix_tmp.n_rows; ++i)
      {
	for (int j = 0; j < coeff_matrix_tmp.n_cols; ++j)
	{
	  coeff_matrix_tmp(i,j) = 1.0 + my_ran();
	}
      }
    }

    {
      for (int i = 0; i < 200; ++i)
      {
	one_iteration_mul(data_matrix_nn, basis_matrix_tmp, coeff_matrix_tmp, 0.2);
      }
      for (int i = 0; i < 200; ++i)
      {
	one_iteration_mul(data_matrix_nn, basis_matrix_tmp, coeff_matrix_tmp, 1);
      }
      for (int i = 0; i < N_iter; ++i)
      {
	if ((p%2) == 0)
	{
	  one_iteration_mul(data_matrix_nn, basis_matrix_tmp, coeff_matrix_tmp, 0.95);
	}
	else
	{
	  double rate = 0.05;
	  one_iteration_add(data_matrix_nn, basis_matrix_tmp, coeff_matrix_tmp, rate);
	}
      }
    }

    double RSS = energy_func(data_matrix, basis_matrix_tmp, coeff_matrix_tmp);

#pragma omp critical
    {
      if (best_E < 0 || best_E > RSS)
      {
	basis_matrix_best 	= basis_matrix_tmp;
	coeff_matrix_best 	= coeff_matrix_tmp;
	best_E 			= RSS;
      }
    }
  }
  return 0;
}

double get_F(const vector<double> &v1, const double x1)
{
  if (x1 < v1[0]) return 0;
  if (x1 > v1.back()) return 1;

  int i_max(0);

  for (int i = 0; i < v1.size(); ++i)
  {
    i_max = i;
    if (v1[i] > x1) break;
  }

  return double(i_max) / double(v1.size());
}

double get_p_from_KS_test(vector<double> & v1, vector<double> &v2)
{
  sort(v1.begin(), v1.end());
  sort(v2.begin(), v2.end());

  vector<double> v12(v1.size() + v2.size());

  copy(v1.begin(), v1.end(), v12.begin());
  copy(v2.begin(), v2.end(), v12.begin() + v1.size());

  if (v12.size() < v1.size())
  {
    exit (1);
  }

  double max_F12(0);

#pragma omp parallel for
  for (int i = 0; i < v12.size(); ++i)
  {
    double tmp = fabs(get_F(v1, v12[i]) - get_F(v2, v12[i]));
#pragma omp critical
    if (tmp > max_F12) max_F12 = tmp;
  }

  double C_alpha = sqrt(v1.size() * v2.size() / double(v1.size() + v2.size())) * max_F12;

  double alpha = 2.0 * exp(-2.0 * C_alpha * C_alpha);

  if (alpha > 1) alpha = 1;

  return alpha;
}

int reorangize_patterns(mat & basis_matrix, mat & coeff_matrix, vector<double> &pattern_sign)
{
  for (int i = 0; i < pattern_sign.size(); ++i)
  {
    for (int j = i; j < pattern_sign.size(); ++j)
    {
      if (pattern_sign[i] < pattern_sign[j])
      {
	double tmp0 = pattern_sign[i];
	pattern_sign[i] = pattern_sign[j];
	pattern_sign[j] = tmp0;

	vec tmp1 = basis_matrix.col(i);
	basis_matrix.col(i) = basis_matrix.col(j);
	basis_matrix.col(j) = tmp1;

	rowvec tmp2 = coeff_matrix.row(i);
	coeff_matrix.row(i) = coeff_matrix.row(j);
	coeff_matrix.row(j) = tmp2;
      }
    }
  }
  return 0;
}



int get_significance(const mat &data_matrix_in, const mat &basis_matrix, 
                     const mat &coeff_matrix, const int N_sample, v_d & pattern_sign)
{
  mat data_matrix = 0.5 * (data_matrix_in + abs(data_matrix_in));

  ///// Significance Test /////

  const int N_order = basis_matrix.n_cols;

  pattern_sign.clear();
  pattern_sign.assign(N_order, 0);

  ///// Generate Random Projection for Comparison ////////

  vector< vector<double> > shu_dist(N_order, vector<double>(0));

#pragma omp parallel for
  for (int t = 0; t < N_sample; ++t)
  {
    my_ran_class my_ran;

    my_ran.set_seed(t);

    mat data_matrix_temp(data_matrix.n_rows, data_matrix.n_cols);

    vector<int> shuffled_idx(data_matrix.n_cols);

    for (int i = 0; i < shuffled_idx.size(); ++i)
    {
      shuffled_idx[i] = i;
    }

    for (int i = 0; i < data_matrix_temp.n_rows; ++i)
    {
      shuffle(shuffled_idx.begin(), shuffled_idx.end(), std::default_random_engine(t));

      for (int j = 0; j < data_matrix_temp.n_cols; ++j)
      {
	data_matrix_temp(i,j) = data_matrix(i, shuffled_idx[j]);
      }
    }

    mat basis_proj_tmp = data_matrix_temp.t() * basis_matrix;

#pragma omp critical
    {
      for (int i = 0; i < N_order; ++i)
      {
	for (int j = 0; j < basis_proj_tmp.n_rows; ++j)
	{
	  shu_dist[i].push_back(basis_proj_tmp(j,i));
	}
      }
    }
  }

#pragma omp parallel for
  for (int i = 0 ; i < shu_dist.size(); ++i)
  {
    sort(shu_dist[i].begin(), shu_dist[i].end());
  }

  vector< vector<double> > data_proj_dist(N_order, vector<double>(0));

  {
    mat basis_proj_tmp = data_matrix.t() * basis_matrix;

    {
      for (int i = 0; i < N_order; ++i)
      {
	for (int j = 0; j < basis_proj_tmp.n_rows; ++j)
	{
	  data_proj_dist[i].push_back(basis_proj_tmp(j,i));
	}
      }
    }
  }

  {
    for (int i = 0; i < N_order; ++i)
    {
      sort(data_proj_dist[i].begin(), data_proj_dist[i].end());
    }
  }

  for (int i = 0; i < N_order; ++i)
  {
    pattern_sign[i] = 1.0 - get_p_from_KS_test(shu_dist[i], data_proj_dist[i]);
  }

  return 0;
}

int print_help(int argc, char ** argv)
{
  string helptag("-h");

  int go(0);

  for (int i = 0; i < argc; ++i)
  {
    if (helptag.compare(argv[i]) == 0)
    {
      go++;
    }
  }

  if (argc == 1) go++;

  if (go)
  {
    cerr << "Options:" << endl;
    cerr << "\t-Ns [int]\t" << "Starting Order" << endl;
    cerr << "\t\tIf the starting order is not defined, it will be 1.\n" << endl;

    cerr << "\t-Ne [int]\t" << "Ending Order" << endl;
    cerr << "\t\tIf the ending order is not valid, e.g. negative or undefined,\n\t\tit will be the same as the starting order\n" << endl;

    cerr << "\t-Na [int]\t" << "Number of attempts for each order" << endl;
    cerr << "\t\tIf this number is not valid or undefined, it will be 1000.\n" << endl;

    cerr << "\t-Ni [int]\t" << "Number of iterations for each attempt" << endl;
    cerr << "\t\tIf this number is not valid or undefined, it will be 1000.\n" << endl;

    cerr << "\t-Nshu [int]\t" << "Number of shuffles for significance" << endl;
    cerr << "\t\tIf this number is not valid or undefined, it will be 100.\n" << endl;

    cerr << "\t-i [filename]\t" << "Filename of the input data" << endl;
    cerr << "\t\tIf this option is missed, the program will terminate\n" << endl;

    cerr << "\t-Nd [int]\t" << "Number of bins for down-sampling" << endl;
    cerr << "\t\tIf this number is not valid or undefined, it will be 1 (for -tr)\n\t\tor 4 (for -tc).\n" << endl;

    cerr << "\t-tc or -tr\tWhether the time axis of the data is along column or row." << endl;
    cerr << "\t\tIf this option is missed, the program will terminate.\n" << endl;

    cerr << "\t-oe [filename]\t" << "Filename of the output for cost function" << endl;
    cerr << "\t-ob [filename]\t" << "Filename of the output of patterns of the optimal order" << endl;
    cerr << "\t-oc [filename]\t" << "Filename of the output of intensities of the patterns" << endl;
    cerr << "\t-os [filename]\t" << "Filename of the output of significance of the patterns" << endl;

    cerr << "\n\t-v\tPrint detail for each order" << endl;
    cerr << "\n\t-S\tStop attempts if the last AIC(t) is larger than the first AIC(1)" << endl;
    cerr << endl;

    cerr << "Examples:" << endl;
    cerr << "\t" << argv[0] << " -v -tc -i nv100.txt -Ns 1 -Ne 20" << endl;
    cerr << "\n\tThese option will make the program perform NMF on nv100.txt from\n\torder 1 to order 20." << endl;

    cerr << "\n\t" << argv[0] << " -v -tc -i nv100.txt -Ns 1 -Ne 50 -oe Cost.txt -ob Patterns.txt -oc Occurrence.txt" << endl;
    cerr << "\n\tThese option will make the program perform NMF on nv100.txt from\n\torder 1 to order 50 and output the result to various files quoted above.\n" << endl;
    
    cerr << "\t" << argv[0] << " -v -tc -i nv100.txt -Ns 20 -ob Patterns.txt -oc Occurrence.txt" << endl;
    cerr << "\n\tThese option will make the program perform NMF on nv100.txt for order 20 ONLY.";
    cerr << "\n\tThe extracted patterns and occurrence are outputted to corresponding files." << endl;
    cerr << endl;
    cerr << "This program was compiled on " << __DATE__ << "." << endl;
    return 1;
  }
  
  return 0;
}
