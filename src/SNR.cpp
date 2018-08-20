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

#include <SNR.hpp>

namespace SNR {

snrConf::snrConf() : KernelConf(), subbandDedispersion(false) {}

snrConf::~snrConf() {}

std::string snrConf::print() const {
  return std::to_string(subbandDedispersion) + " " + isa::OpenCL::KernelConf::print();
}

void readTunedSNRConf(tunedSNRConf & tunedSNR, const std::string & snrFilename) {
  unsigned int nrDMs = 0;
  unsigned int nrSamples = 0;
  std::string temp;
  std::string deviceName;
  SNR::snrConf * parameters = 0;
  std::ifstream snrFile;

  snrFile.open(snrFilename);
  if ( !snrFile ) {
    throw AstroData::FileError("Impossible to open " + snrFilename );
  }
  while ( ! snrFile.eof() ) {
    unsigned int splitPoint = 0;

    std::getline(snrFile, temp);
    if ( ! std::isalpha(temp[0]) ) {
      continue;
    }
    parameters = new SNR::snrConf();

    splitPoint = temp.find(" ");
    deviceName = temp.substr(0, splitPoint);
    temp = temp.substr(splitPoint + 1);
    splitPoint = temp.find(" ");
    nrDMs = isa::utils::castToType< std::string, unsigned int >(temp.substr(0, splitPoint));
    temp = temp.substr(splitPoint + 1);
    splitPoint = temp.find(" ");
    nrSamples = isa::utils::castToType< std::string, unsigned int >(temp.substr(0, splitPoint));
    temp = temp.substr(splitPoint + 1);
    splitPoint = temp.find(" ");
    parameters->setSubbandDedispersion(isa::utils::castToType< std::string, bool >(temp.substr(0, splitPoint)));
    temp = temp.substr(splitPoint + 1);
    splitPoint = temp.find(" ");
    parameters->setNrThreadsD0(isa::utils::castToType< std::string, unsigned int >(temp.substr(0, splitPoint)));
    temp = temp.substr(splitPoint + 1);
    splitPoint = temp.find(" ");
    parameters->setNrThreadsD1(isa::utils::castToType< std::string, unsigned int >(temp.substr(0, splitPoint)));
    temp = temp.substr(splitPoint + 1);
    splitPoint = temp.find(" ");
    parameters->setNrThreadsD2(isa::utils::castToType< std::string, unsigned int >(temp.substr(0, splitPoint)));
    temp = temp.substr(splitPoint + 1);
    splitPoint = temp.find(" ");
    parameters->setNrItemsD0(isa::utils::castToType< std::string, unsigned int >(temp.substr(0, splitPoint)));
    temp = temp.substr(splitPoint + 1);
    splitPoint = temp.find(" ");
    parameters->setNrItemsD1(isa::utils::castToType< std::string, unsigned int >(temp.substr(0, splitPoint)));
    temp = temp.substr(splitPoint + 1);
    parameters->setNrItemsD2(isa::utils::castToType< std::string, unsigned int >(temp));

    if ( tunedSNR.count(deviceName) == 0 ) {
      std::map< unsigned int, std::map< unsigned int, SNR::snrConf * > * > * externalContainer = new std::map< unsigned int, std::map< unsigned int, SNR::snrConf * > * >();
      std::map< unsigned int, SNR::snrConf * > * internalContainer = new std::map< unsigned int, SNR::snrConf * >();

      internalContainer->insert(std::make_pair(nrSamples, parameters));
      externalContainer->insert(std::make_pair(nrDMs, internalContainer));
      tunedSNR.insert(std::make_pair(deviceName, externalContainer));
    } else if ( tunedSNR.at(deviceName)->count(nrDMs) == 0 ) {
      std::map< unsigned int, SNR::snrConf * > * internalContainer = new std::map< unsigned int, SNR::snrConf * >();

      internalContainer->insert(std::make_pair(nrSamples, parameters));
      tunedSNR.at(deviceName)->insert(std::make_pair(nrDMs, internalContainer));
    } else {
      tunedSNR.at(deviceName)->at(nrDMs)->insert(std::make_pair(nrSamples, parameters));
    }
  }
  snrFile.close();
}

} // SNR

