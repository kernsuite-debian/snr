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

#include <iostream>
#include <string>
#include <vector>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <ctime>

#include <configuration.hpp>

#include <ArgumentList.hpp>
#include <Observation.hpp>
#include <InitializeOpenCL.hpp>
#include <Kernel.hpp>
#include <utils.hpp>
#include <SNR.hpp>
#include <Stats.hpp>


int main(int argc, char *argv[]) {
  bool printCode = false;
  bool printResults = false;
  bool DMsSamples = false;
  unsigned int padding = 0;
  unsigned int clPlatformID = 0;
  unsigned int clDeviceID = 0;
  uint64_t wrongSamples = 0;
  uint64_t wrongPositions = 0;
  AstroData::Observation observation;
  SNR::snrConf conf;

  try {
    isa::utils::ArgumentList args(argc, argv);
    DMsSamples = args.getSwitch("-dms_samples");
    bool samplesDMs = args.getSwitch("-samples_dms");
    if ( (DMsSamples && samplesDMs) || (!DMsSamples && !samplesDMs) ) {
      std::cerr << "-dms_samples and -samples_dms are mutually exclusive." << std::endl;
      return 1;
    }
    printCode = args.getSwitch("-print_code");
    printResults = args.getSwitch("-print_results");
    clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
    clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");
    padding = args.getSwitchArgument< unsigned int >("-padding");
    conf.setNrThreadsD0(args.getSwitchArgument< unsigned int >("-threadsD0"));
    conf.setNrItemsD0(args.getSwitchArgument< unsigned int >("-itemsD0"));
    conf.setSubbandDedispersion(args.getSwitch("-subband"));
    observation.setNrSynthesizedBeams(args.getSwitchArgument< unsigned int >("-beams"));
    observation.setNrSamplesPerBatch(args.getSwitchArgument< unsigned int >("-samples"));
    if ( conf.getSubbandDedispersion() ) {
      observation.setDMRange(args.getSwitchArgument< unsigned int >("-subbanding_dms"), 0.0f, 0.0f, true);
    } else {
      observation.setDMRange(1, 0.0f, 0.0f, true);
    }
    observation.setDMRange(args.getSwitchArgument< unsigned int >("-dms"), 0.0, 0.0);
  } catch  ( isa::utils::SwitchNotFound & err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  } catch ( std::exception &err ) {
    std::cerr << "Usage: " << argv[0] << " [-dms_samples | -samples_dms] [-print_code] [-print_results] -opencl_platform ... -opencl_device ... -padding ... -threadsD0 ... -itemsD0 ... [-subband] -beams ... -dms ... -samples ..." << std::endl;
    std::cerr << "\t -subband : -subbanding_dms ..." << std::endl;
    return 1;
  }

  // Initialize OpenCL
  cl::Context * clContext = new cl::Context();
  std::vector< cl::Platform > * clPlatforms = new std::vector< cl::Platform >();
  std::vector< cl::Device > * clDevices = new std::vector< cl::Device >();
  std::vector< std::vector< cl::CommandQueue > > * clQueues = new std::vector< std::vector < cl::CommandQueue > >();

  isa::OpenCL::initializeOpenCL(clPlatformID, 1, clPlatforms, clContext, clDevices, clQueues);

  // Allocate memory
  std::vector< inputDataType > input;
  std::vector< float > outputSNR;
  std::vector< unsigned int > outputSample;
  cl::Buffer input_d, outputSNR_d, outputSample_d;
  if ( DMsSamples ) {
    input.resize(observation.getNrSynthesizedBeams() * observation.getNrDMs(true) * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(inputDataType)));
  } else {
    input.resize(observation.getNrSynthesizedBeams() * observation.getNrSamplesPerBatch() * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(inputDataType)));
  }
  outputSNR.resize(observation.getNrSynthesizedBeams() * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(float)));
  outputSample.resize(observation.getNrSynthesizedBeams() * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(unsigned int)));
  try {
    input_d = cl::Buffer(*clContext, CL_MEM_READ_WRITE, input.size() * sizeof(inputDataType), 0, 0);
    outputSNR_d = cl::Buffer(*clContext, CL_MEM_WRITE_ONLY, outputSNR.size() * sizeof(float), 0, 0);
    outputSample_d = cl::Buffer(*clContext, CL_MEM_WRITE_ONLY, outputSample.size() * sizeof(unsigned int), 0, 0);
  } catch ( cl::Error &err ) {
    std::cerr << "OpenCL error allocating memory: " << std::to_string(err.err()) << "." << std::endl;
    return 1;
  }

  // Generate test data
  std::vector< unsigned int > maxSample(observation.getNrSynthesizedBeams() * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(unsigned int)));

  srand(time(0));
  for ( auto item = maxSample.begin(); item != maxSample.end(); ++item ) {
    *item = rand() % observation.getNrSamplesPerBatch();
  }
  for ( unsigned int beam = 0; beam < observation.getNrSynthesizedBeams(); beam++ ) {
    if ( printResults ) {
      std::cout << "Beam: " << beam << std::endl;
    }
    if ( DMsSamples ) {
      for ( unsigned int subbandDM = 0; subbandDM < observation.getNrDMs(true); subbandDM++ ) {
        for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
          if ( printResults ) {
            std::cout << "DM: " << (subbandDM * observation.getNrDMs()) + dm << " -- ";
          }
          for ( unsigned int sample = 0; sample < observation.getNrSamplesPerBatch(); sample++ ) {
            if ( sample == maxSample.at((beam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(unsigned int))) + (subbandDM * observation.getNrDMs()) + dm) ) {
              input[(beam * observation.getNrDMs(true) * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(inputDataType))) + (subbandDM * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(inputDataType))) + (dm * observation.getNrSamplesPerBatch(false, padding / sizeof(inputDataType))) + sample] = static_cast< inputDataType >(10 + (rand() % 10));
            } else {
              input[(beam * observation.getNrDMs(true) * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(inputDataType))) + (subbandDM * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(inputDataType))) + (dm * observation.getNrSamplesPerBatch(false, padding / sizeof(inputDataType))) + sample] = static_cast< inputDataType >(rand() % 10);
            }
            if ( printResults ) {
              std::cout << input[(beam * observation.getNrDMs(true) * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(inputDataType))) + (subbandDM * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(inputDataType))) + (dm * observation.getNrSamplesPerBatch(false, padding / sizeof(inputDataType))) + sample] << " ";
            }
          }
          if ( printResults ) {
            std::cout << std::endl;
          }
        }
      }
    } else {
      for ( unsigned int sample = 0; sample < observation.getNrSamplesPerBatch(); sample++ ) {
        if ( printResults ) {
          std::cout << "Sample: " << sample << " -- ";
        }
        for ( unsigned int subbandDM = 0; subbandDM < observation.getNrDMs(true); subbandDM++ ) {
          for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
            if ( sample == maxSample.at((beam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(unsigned int))) + (subbandDM * observation.getNrDMs()) + dm) ) {
              input[(beam * observation.getNrSamplesPerBatch() * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(inputDataType))) + (sample * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(inputDataType))) + (subbandDM * observation.getNrDMs(false, padding / sizeof(inputDataType))) + dm] = static_cast< inputDataType >(10 + (rand() % 10));
            } else {
              input[(beam * observation.getNrSamplesPerBatch() * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(inputDataType))) + (sample * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(inputDataType))) + (subbandDM * observation.getNrDMs(false, padding / sizeof(inputDataType))) + dm] = static_cast< inputDataType >(rand() % 10);
            }
            if ( printResults ) {
              std::cout << input[(beam * observation.getNrSamplesPerBatch() * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(inputDataType))) + (sample * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(inputDataType))) + (subbandDM * observation.getNrDMs(false, padding / sizeof(inputDataType))) + dm] << " ";
            }
          }
          if ( printResults ) {
            std::cout << std::endl;
          }
        }
      }
    }
  }
  if ( printResults ) {
    std::cout << std::endl;
  }

  // Copy data structures to device
  try {
    clQueues->at(clDeviceID)[0].enqueueWriteBuffer(input_d, CL_FALSE, 0, input.size() * sizeof(inputDataType), reinterpret_cast< void * >(input.data()));
  } catch ( cl::Error &err ) {
    std::cerr << "OpenCL error H2D transfer: " << std::to_string(err.err()) << "." << std::endl;
    return 1;
  }

  // Generate kernel
  cl::Kernel * kernel;
  std::string * code;
  if ( DMsSamples ) {
    code = SNR::getSNRDMsSamplesOpenCL< inputDataType >(conf, inputDataName, observation, observation.getNrSamplesPerBatch(), padding);
  } else {
    code = SNR::getSNRSamplesDMsOpenCL< inputDataType >(conf, inputDataName, observation, observation.getNrSamplesPerBatch(), padding);
  }
  if ( printCode ) {
    std::cout << *code << std::endl;
  }

  try {
    if ( DMsSamples ) {
      kernel = isa::OpenCL::compile("snrDMsSamples" + std::to_string(observation.getNrSamplesPerBatch()), *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(clDeviceID));
    } else {
      kernel = isa::OpenCL::compile("snrSamplesDMs" + std::to_string(observation.getNrDMs(true) * observation.getNrDMs()), *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(clDeviceID));
    }
  } catch ( isa::OpenCL::OpenCLError & err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }

  // Run OpenCL kernel and CPU control
  std::vector< isa::utils::Stats< inputDataType > > control(observation.getNrSynthesizedBeams() * observation.getNrDMs(true) * observation.getNrDMs());
  try {
    cl::NDRange global;
    cl::NDRange local;

    if ( DMsSamples ) {
      global = cl::NDRange(conf.getNrThreadsD0(), observation.getNrDMs(true) * observation.getNrDMs(), observation.getNrSynthesizedBeams());
      local = cl::NDRange(conf.getNrThreadsD0(), 1, 1);
    } else {
      global = cl::NDRange((observation.getNrDMs(true) * observation.getNrDMs()) / conf.getNrItemsD0(), observation.getNrSynthesizedBeams());
      local = cl::NDRange(conf.getNrThreadsD0(), 1);
    }

    kernel->setArg(0, input_d);
    kernel->setArg(1, outputSNR_d);
    kernel->setArg(2, outputSample_d);

    clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local, 0, 0);
    clQueues->at(clDeviceID)[0].enqueueReadBuffer(outputSNR_d, CL_TRUE, 0, outputSNR.size() * sizeof(float), reinterpret_cast< void * >(outputSNR.data()));
    clQueues->at(clDeviceID)[0].enqueueReadBuffer(outputSample_d, CL_TRUE, 0, outputSample.size() * sizeof(unsigned int), reinterpret_cast< void * >(outputSample.data()));
  } catch ( cl::Error & err ) {
    std::cerr << "OpenCL error: " << std::to_string(err.err()) << "." << std::endl;
    return 1;
  }
  for ( unsigned int beam = 0; beam < observation.getNrSynthesizedBeams(); beam++ ) {
    for ( unsigned int subbandDM = 0; subbandDM < observation.getNrDMs(true); subbandDM++ ) {
      for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
        control[(beam * observation.getNrDMs(true) * observation.getNrDMs()) + (subbandDM * observation.getNrDMs()) + dm] = isa::utils::Stats< inputDataType >();
      }
    }
    if ( DMsSamples ) {
      for ( unsigned int subbandDM = 0; subbandDM < observation.getNrDMs(true); subbandDM++ ) {
        for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
          for ( unsigned int sample = 0; sample < observation.getNrSamplesPerBatch(); sample++ ) {
            control[(beam * observation.getNrDMs(true) * observation.getNrDMs()) + (subbandDM * observation.getNrDMs()) + dm].addElement(input[(beam * observation.getNrDMs(true) * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(inputDataType))) + (subbandDM * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(inputDataType))) + (dm * observation.getNrSamplesPerBatch(false, padding / sizeof(inputDataType))) + sample]);
          }
        }
      }
    } else {
      for ( unsigned int sample = 0; sample < observation.getNrSamplesPerBatch(); sample++ ) {
        for ( unsigned int subbandDM = 0; subbandDM < observation.getNrDMs(true); subbandDM++ ) {
          for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
            control[(beam * observation.getNrDMs(true) * observation.getNrDMs()) + (subbandDM * observation.getNrDMs()) + dm].addElement(input[(beam * observation.getNrSamplesPerBatch() * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(inputDataType))) + (sample * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(inputDataType))) + (subbandDM * observation.getNrDMs(false, padding / sizeof(inputDataType))) + dm]);
          }
        }
      }
    }
  }

  for ( unsigned int beam = 0; beam < observation.getNrSynthesizedBeams(); beam++ ) {
    for ( unsigned int subbandDM = 0; subbandDM < observation.getNrDMs(true); subbandDM++ ) {
      for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
        if ( !isa::utils::same(outputSNR[(beam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(float))) + (subbandDM * observation.getNrDMs()) + dm], static_cast<float>((control[(beam * observation.getNrDMs(true) * observation.getNrDMs()) + (subbandDM * observation.getNrDMs()) + dm].getMax() - control[(beam * observation.getNrDMs(true) * observation.getNrDMs()) + (subbandDM * observation.getNrDMs()) + dm].getMean()) / control[(beam * observation.getNrDMs(true) * observation.getNrDMs()) + (subbandDM * observation.getNrDMs()) + dm].getStandardDeviation()), static_cast<float>(1e-2)) ) {
          wrongSamples++;
        }
        if ( outputSample.at((beam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(unsigned int))) + (subbandDM * observation.getNrDMs()) + dm) != maxSample.at((beam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(unsigned int))) + (subbandDM * observation.getNrDMs()) + dm) ) {
          wrongPositions++;
        }
      }
    }
  }
  
  if ( printResults ) {
    for ( unsigned int beam = 0; beam < observation.getNrSynthesizedBeams(); beam++ ) {
      std::cout << "Beam: " << beam << std::endl;
      for ( unsigned int subbandDM = 0; subbandDM < observation.getNrDMs(true); subbandDM++ ) {
        for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
          std::cout << outputSNR[(beam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(float))) + (subbandDM * observation.getNrDMs()) + dm] << "," << (control[(beam * observation.getNrDMs(true) * observation.getNrDMs()) + (subbandDM * observation.getNrDMs()) + dm].getMax() - control[(beam * observation.getNrDMs(true) * observation.getNrDMs()) + (subbandDM * observation.getNrDMs()) + dm].getMean()) / control[(beam * observation.getNrDMs(true) * observation.getNrDMs()) + (subbandDM * observation.getNrDMs()) + dm].getStandardDeviation() << " ; ";
          std::cout << outputSample.at((beam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(unsigned int))) + (subbandDM * observation.getNrDMs()) + dm) << "," << maxSample.at((beam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(unsigned int))) + (subbandDM * observation.getNrDMs()) + dm) << "  ";
        }
        std::cout << std::endl;
        }
      }
    std::cout << std::endl;
  }

  if ( wrongSamples > 0 ) {
    std::cout << "Wrong samples: " << wrongSamples << " (" << (wrongSamples * 100.0) / static_cast< uint64_t >(observation.getNrSynthesizedBeams() * observation.getNrDMs(true) * observation.getNrDMs()) << "%)." << std::endl;
  } else if ( wrongPositions > 0 ) {
    std::cout << "Wrong positions: " << wrongPositions << " (" << (wrongPositions * 100.0) / static_cast< uint64_t >(observation.getNrSynthesizedBeams() * observation.getNrDMs(true) * observation.getNrDMs()) << "%)." << std::endl;
  } else {
    std::cout << "TEST PASSED." << std::endl;
  }

  return 0;
}

