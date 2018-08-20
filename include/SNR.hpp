// Copyright 2017 Netherlands Institute for Radio Astronomy (ASTRON)
// Copyright 2017 Netherlands eScience Center
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vector>
#include <string>
#include <cmath>
#include <map>
#include <fstream>

#include <Kernel.hpp>
#include <Observation.hpp>
#include <Platform.hpp>
#include <utils.hpp>

#pragma once

namespace SNR {

class snrConf : public isa::OpenCL::KernelConf {
public:
  snrConf();
  ~snrConf();
  // Get
  bool getSubbandDedispersion() const;
  // Set
  void setSubbandDedispersion(bool subband);
  // utils
  std::string print() const;

private:
  bool subbandDedispersion;
};

typedef std::map<std::string, std::map<unsigned int, std::map<unsigned int, SNR::snrConf *> *> *> tunedSNRConf;

// OpenCL SNR
template<typename T> std::string * getSNRDMsSamplesOpenCL(const snrConf & conf, const std::string & dataName, const AstroData::Observation & observation, const unsigned int nrSamples, const unsigned int padding);
template<typename T> std::string * getSNRSamplesDMsOpenCL(const snrConf & conf, const std::string & dataName, const AstroData::Observation & observation, const unsigned int nrSamples, const unsigned int padding);
// Read configuration files
void readTunedSNRConf(tunedSNRConf & tunedSNR, const std::string & snrFilename);


// Implementations
inline bool snrConf::getSubbandDedispersion() const {
  return subbandDedispersion;
}

inline void snrConf::setSubbandDedispersion(bool subband) {
  subbandDedispersion = subband;
}

template<typename T> std::string * getSNRDMsSamplesOpenCL(const snrConf & conf, const std::string & dataName, const AstroData::Observation & observation, const unsigned int nrSamples, const unsigned int padding) {
  unsigned int nrDMs = 0;
  std::string * code = new std::string();

  if ( conf.getSubbandDedispersion() ) {
    nrDMs = observation.getNrDMs(true) * observation.getNrDMs();
  } else {
    nrDMs = observation.getNrDMs();
  }
  // Begin kernel's template
  *code = "__kernel void snrDMsSamples" + std::to_string(nrSamples) + "(__global const " + dataName + " * const restrict input, __global float * const restrict outputSNR, __global unsigned int * const restrict outputSample) {\n"
    "unsigned int dm = get_group_id(1);\n"
    "unsigned int beam = get_group_id(2);\n"
    "float delta = 0.0f;\n"
    "__local float reductionCOU[" + std::to_string(isa::utils::pad(conf.getNrThreadsD0(), padding / sizeof(float))) + "];\n"
    "__local " + dataName + " reductionMAX[" + std::to_string(isa::utils::pad(conf.getNrThreadsD0(), padding / sizeof(T))) + "];\n"
    "__local unsigned int reductionSAM[" + std::to_string(isa::utils::pad(conf.getNrThreadsD0(), padding / sizeof(unsigned int))) + "];\n"
    "__local float reductionMEA[" + std::to_string(isa::utils::pad(conf.getNrThreadsD0(), padding / sizeof(float))) + "];\n"
    "__local float reductionVAR[" + std::to_string(isa::utils::pad(conf.getNrThreadsD0(), padding / sizeof(float))) + "];\n"
    "<%DEF%>"
    "\n"
    "// Compute phase\n"
    "for ( unsigned int sample = get_local_id(0) + " + std::to_string(conf.getNrThreadsD0() * conf.getNrItemsD0()) + "; sample < " + std::to_string(nrSamples) + "; sample += " + std::to_string(conf.getNrThreadsD0() * conf.getNrItemsD0()) + " ) {\n"
    + dataName + " item = 0;\n"
    "<%COMPUTE%>"
    "}\n"
    "// In-thread reduce\n"
    "<%REDUCE%>"
    "// Local memory store\n"
    "reductionCOU[get_local_id(0)] = counter0;\n"
    "reductionMAX[get_local_id(0)] = max0;\n"
    "reductionSAM[get_local_id(0)] = maxSample0;\n"
    "reductionMEA[get_local_id(0)] = mean0;\n"
    "reductionVAR[get_local_id(0)] = variance0;\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "// Reduce phase\n"
    "unsigned int threshold = " + std::to_string(conf.getNrThreadsD0() / 2) + ";\n"
    "for ( unsigned int sample = get_local_id(0); threshold > 0; threshold /= 2 ) {\n"
    "if ( sample < threshold ) {\n"
    "delta = reductionMEA[sample + threshold] - mean0;\n"
    "counter0 += reductionCOU[sample + threshold];\n"
    "mean0 = ((reductionCOU[sample] * mean0) + (reductionCOU[sample + threshold] * reductionMEA[sample + threshold])) / counter0;\n"
    "variance0 += reductionVAR[sample + threshold] + ((delta * delta) * ((reductionCOU[sample] * reductionCOU[sample + threshold]) / counter0));\n"
    "if ( reductionMAX[sample + threshold] - max0 > 0.0f ) {\n"
    "max0 = reductionMAX[sample + threshold];\n"
    "maxSample0 = reductionSAM[sample + threshold];\n"
    "}\n"
    "reductionCOU[sample] = counter0;\n"
    "reductionMAX[sample] = max0;\n"
    "reductionSAM[sample] = maxSample0;\n"
    "reductionMEA[sample] = mean0;\n"
    "reductionVAR[sample] = variance0;\n"
    "}\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "}\n"
    "// Store\n"
    "if ( get_local_id(0) == 0 ) {\n"
    "outputSNR[(beam * " + std::to_string(isa::utils::pad(nrDMs, padding / sizeof(float))) + ") + dm] = (max0 - mean0) / native_sqrt(variance0 * " + std::to_string(1.0f / (nrSamples - 1)) + "f);\n"
    "outputSample[(beam * " + std::to_string(isa::utils::pad(nrDMs, padding / sizeof(unsigned int))) + ") + dm] = maxSample0;\n"
    "}\n"
    "}\n";
  std::string def_sTemplate = "float counter<%NUM%> = 1.0f;\n"
    "unsigned int maxSample<%NUM%> = get_local_id(0) + <%OFFSET%>;\n"
    + dataName + " max<%NUM%> = input[(beam * " + std::to_string(nrDMs * isa::utils::pad(nrSamples, padding / sizeof(T))) + ") + (dm * " + std::to_string(isa::utils::pad(nrSamples, padding / sizeof(T))) + ") + (get_local_id(0) + <%OFFSET%>)];\n"
    "float variance<%NUM%> = 0.0f;\n"
    "float mean<%NUM%> = max<%NUM%>;\n";
  std::string compute_sTemplate;
  if ( (nrSamples % (conf.getNrThreadsD0() * conf.getNrItemsD0())) != 0 ) {
    compute_sTemplate += "if ( (sample + <%OFFSET%>) < " + std::to_string(nrSamples) + " ) {\n";
  }
  compute_sTemplate += "item = input[(beam * " + std::to_string(nrDMs * isa::utils::pad(nrSamples, padding / sizeof(T))) + ") + (dm * " + std::to_string(isa::utils::pad(nrSamples, padding / sizeof(T))) + ") + (sample + <%OFFSET%>)];\n"
    "counter<%NUM%> += 1.0f;\n"
    "delta = item - mean<%NUM%>;\n"
    "mean<%NUM%> += delta / counter<%NUM%>;\n"
    "variance<%NUM%> += delta * (item - mean<%NUM%>);\n"
    "if ( item - max<%NUM%> > 0.0f ) {\n"
    "max<%NUM%> = item;\n"
    "maxSample<%NUM%> = sample + <%OFFSET%>;\n"
    "}\n";
  if ( (nrSamples % (conf.getNrThreadsD0() * conf.getNrItemsD0())) != 0 ) {
    compute_sTemplate += "}\n";
  }
  std::string reduce_sTemplate = "delta = mean<%NUM%> - mean0;\n"
    "counter0 += counter<%NUM%>;\n"
    "mean0 = (((counter0 - counter<%NUM%>) * mean0) + (counter<%NUM%> * mean<%NUM%>)) / counter0;\n"
    "variance0 += variance<%NUM%> + ((delta * delta) * (((counter0 - counter<%NUM%>) * counter<%NUM%>) / counter0));\n"
    "if ( max<%NUM%> - max0 > 0.0f ) {\n"
    "max0 = max<%NUM%>;\n"
    "maxSample0 = maxSample<%NUM%>;\n"
    "}\n";
  // End kernel's template

  std::string * def_s = new std::string();
  std::string * compute_s = new std::string();
  std::string * reduce_s = new std::string();

  for ( unsigned int sample = 0; sample < conf.getNrItemsD0(); sample++ ) {
    std::string sample_s = std::to_string(sample);
    std::string offset_s = std::to_string(conf.getNrThreadsD0() * sample);
    std::string * temp = 0;

    temp = isa::utils::replace(&def_sTemplate, "<%NUM%>", sample_s);
    if ( sample == 0 ) {
      std::string empty_s("");
      temp = isa::utils::replace(temp, " + <%OFFSET%>", empty_s, true);
    } else {
      temp = isa::utils::replace(temp, "<%OFFSET%>", offset_s, true);
    }
    def_s->append(*temp);
    delete temp;
    temp = isa::utils::replace(&compute_sTemplate, "<%NUM%>", sample_s);
    if ( sample == 0 ) {
      std::string empty_s("");
      temp = isa::utils::replace(temp, " + <%OFFSET%>", empty_s, true);
    } else {
      temp = isa::utils::replace(temp, "<%OFFSET%>", offset_s, true);
    }
    compute_s->append(*temp);
    delete temp;
    if ( sample == 0 ) {
      continue;
    }
    temp = isa::utils::replace(&reduce_sTemplate, "<%NUM%>", sample_s);
    reduce_s->append(*temp);
    delete temp;
  }

  code = isa::utils::replace(code, "<%DEF%>", *def_s, true);
  code = isa::utils::replace(code, "<%COMPUTE%>", *compute_s, true);
  code = isa::utils::replace(code, "<%REDUCE%>", *reduce_s, true);
  delete def_s;
  delete compute_s;
  delete reduce_s;

  return code;
}

template<typename T> std::string * getSNRSamplesDMsOpenCL(const snrConf & conf, const std::string & dataName, const AstroData::Observation & observation, const unsigned int nrSamples, const unsigned int padding) {
  unsigned int nrDMs = 0;
  std::string * code = new std::string();

  if ( conf.getSubbandDedispersion() ) {
    nrDMs = observation.getNrDMs(true) * observation.getNrDMs();
  } else {
    nrDMs = observation.getNrDMs();
  }
  // Begin kernel's template
  *code = "__kernel void snrSamplesDMs" + std::to_string(nrDMs) + "(__global const " + dataName + " * const restrict input, __global float * const restrict outputSNR, __global unsigned int * const restrict outputSample) {\n"
    "unsigned int dm = (get_group_id(0) * " + std::to_string(conf.getNrThreadsD0() * conf.getNrItemsD0()) + ") + get_local_id(0);\n"
    "unsigned int beam = get_group_id(1);\n"
    "float delta = 0.0f;\n"
    "<%DEF%>"
    "\n"
    "for ( unsigned int sample = 1; sample < " + std::to_string(nrSamples) + "; sample++ ) {\n"
    + dataName + " item = 0;\n"
    "<%COMPUTE%>"
    "}\n"
    "<%STORE%>"
    "}\n";
  std::string def_sTemplate = "float counter<%NUM%> = 1.0f;\n"
    + dataName + " max<%NUM%> = input[dm + <%OFFSET%>];\n"
    "unsigned int maxSample<%NUM%> = 0;\n"
    "float variance<%NUM%> = 0.0f;\n"
    "float mean<%NUM%> = max<%NUM%>;\n";
  std::string compute_sTemplate = "item = input[(beam * " + std::to_string(nrSamples * isa::utils::pad(nrDMs, padding / sizeof(T))) + ") + (sample * " + std::to_string(isa::utils::pad(nrDMs, padding / sizeof(T))) + ")  + (dm + <%OFFSET%>)];\n"
    "counter<%NUM%> += 1.0f;\n"
    "delta = item - mean<%NUM%>;\n"
    "mean<%NUM%> += delta / counter<%NUM%>;\n"
    "variance<%NUM%> += delta * (item - mean<%NUM%>);\n"
    "if ( item > max<%NUM%> ) {\n"
    "max<%NUM%> = item;\n"
    "maxSample<%NUM%> = sample;\n"
    "}\n";
  std::string store_sTemplate = "outputSNR[(beam * " + std::to_string(isa::utils::pad(nrDMs, padding / sizeof(float))) + ") + dm + <%OFFSET%>] = (max<%NUM%> - mean<%NUM%>) / native_sqrt(variance<%NUM%> * " + std::to_string(1.0f / (observation.getNrSamplesPerBatch() - 1)) + "f);\n";
  // End kernel's template

  std::string * def_s = new std::string();
  std::string * compute_s = new std::string();
  std::string * store_s = new std::string();

  for ( unsigned int dm = 0; dm < conf.getNrItemsD0(); dm++ ) {
    std::string dm_s = std::to_string(dm);
    std::string offset_s = std::to_string(conf.getNrThreadsD0() * dm);
    std::string * temp = 0;

    temp = isa::utils::replace(&def_sTemplate, "<%NUM%>", dm_s);
    if ( dm == 0 ) {
      std::string empty_s("");
      temp = isa::utils::replace(temp, " + <%OFFSET%>", empty_s, true);
    } else {
      temp = isa::utils::replace(temp, "<%OFFSET%>", offset_s, true);
    }
    def_s->append(*temp);
    delete temp;
    temp = isa::utils::replace(&compute_sTemplate, "<%NUM%>", dm_s);
    if ( dm == 0 ) {
      std::string empty_s("");
      temp = isa::utils::replace(temp, " + <%OFFSET%>", empty_s, true);
    } else {
      temp = isa::utils::replace(temp, "<%OFFSET%>", offset_s, true);
    }
    compute_s->append(*temp);
    delete temp;
    temp = isa::utils::replace(&store_sTemplate, "<%NUM%>", dm_s);
    if ( dm == 0 ) {
      std::string empty_s("");
      temp = isa::utils::replace(temp, " + <%OFFSET%>", empty_s, true);
    } else {
      temp = isa::utils::replace(temp, "<%OFFSET%>", offset_s, true);
    }
    store_s->append(*temp);
    delete temp;
  }

  code = isa::utils::replace(code, "<%DEF%>", *def_s, true);
  code = isa::utils::replace(code, "<%COMPUTE%>", *compute_s, true);
  code = isa::utils::replace(code, "<%STORE%>", *store_s, true);
  delete def_s;
  delete compute_s;
  delete store_s;

  return code;
}

} // SNR

