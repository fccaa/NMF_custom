
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <random>
#include <algorithm>

#include <armadillo>

using namespace arma;

using namespace std;

typedef vector<double>	v_d;
typedef vector<v_d>	vv_d;

class my_ran_class
{
  ranlux48 generator;
  uniform_int_distribution<int> uniform_int;
  uniform_real_distribution<double> uniform_double;

  public:
  int operator () (int i)
  {
    return uniform_int(generator) % i;
    //return rand() % i;
  }
  double operator () ()
  {
    return uniform_double(generator);
    //return (rand() % 1000) / 1000.0;
  }
  int set_seed(int i)
  {
    generator.seed(i);

    return 0;
  }
  my_ran_class()
  {
    //srand(0);
  }
};

int shuffle(vector<int> & input, class my_ran_class & ran)
{
  for (int i = 0; i < input.size(); ++i)
  {
    input[i] = i;
  }

  vector<int> input_tmp(input.size(), 0);
  copy(input.begin(), input.end(), input_tmp.begin());

  input.clear();

  while (input_tmp.size())
  {
    int i = ran(int(input_tmp.size()));

    input.push_back(input_tmp[i]);
    input_tmp.erase(input_tmp.begin() + i);
  }

  return 0;
}

int get_data_pattern(v_d &, vv_d &, string &);

int get_data_signi(v_d &, string &);

int main(int argc, char * argv[])
{
  string in_label;

  {
    in_label = argv[1];
  }

  vector<string> in_files;

  /*
     The following for loop is to scan data file in the directory ../../ .
     The names of those files have a format.
     {String}_{A single character}.txt
     */

  for (int i = 0; i < 20; ++i)
  {
    char tmp_c[1000];
    string tmp;
    if (i)
    {
      sprintf(tmp_c, "../../%s_%c.txt", in_label.c_str(), 64+i);
      tmp = tmp_c;
    }
    
    ifstream test;
    test.open(tmp, ifstream::in);
    if (test.good())
    {
      in_files.push_back(tmp);
    }
    test.close();
  }

  vector<mat> all_data;

  for (int i = 0; i < in_files.size(); ++i)
  {
    v_d t_stamps;
    vv_d data;
    get_data_pattern(t_stamps, data, in_files[i]);

    mat new_data(data[0].size(), data.size());

    for (int j = 0; j < data.size(); ++j)
    {
      for (int k = 0; k < data[j].size(); ++k)
      {
	new_data(k,j) = data[j][k];
      }
    }

    all_data.push_back(new_data);
  }

  for (int i = 0; i < all_data.size(); ++i)
  {
    mat all_data12 = join_rows(all_data[0], all_data[i]);

    mat cov12 = cov(all_data12.t(), all_data12.t());

    vec mean1 = mean(all_data[0], 1);
    vec mean2 = mean(all_data[i], 1);

    vec eigval; mat eigvec;

    eig_sym(eigval, eigvec, cov12);

    vec diff = eigvec.t() * (mean1 - mean2) ;
    diff = diff % diff;
    diff = diff / eigval;
    diff = sort(diff, "descend");

    double dist(0);

    for (int i = 0; i < 10; ++i)
    {
      dist += diff[i];
    }

    
    dist = sqrt(dist);

    cout << i << "\t" << dist << endl;
  }


  return EXIT_SUCCESS;
}

int get_data_pattern(v_d & data_t, vv_d & data_signal, string & data_file)
{
  data_t.clear();
  data_signal.clear();

  fstream fin;
  fin.open(data_file);

  while (fin.good())
  {
    string str;
    getline(fin, str);

    stringstream sstr_tmp(str);

    {
      char tmp;
      sstr_tmp >> tmp;
      if (tmp == '#') continue;
    }

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

int get_data_signi(v_d & data_signi, string & data_file)
{
  data_signi.clear();

  fstream fin;
  fin.open(data_file);

  while (fin.good())
  {
    string str;
    getline(fin, str);

    stringstream sstr_tmp(str);

    {
      char tmp;
      sstr_tmp >> tmp;
      if (tmp == '#') continue;
    }

    stringstream sstr(str);

    if (fin.eof())
    {
      break;
    }

    double t_temp(0.0);

    sstr >> t_temp;

    double temp(0.0);

    sstr >> temp;

    data_signi.push_back(temp);
  };

  fin.close();

  return 0;
}
