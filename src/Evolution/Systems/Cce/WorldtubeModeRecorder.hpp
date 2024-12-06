// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "DataStructures/ComplexModalVector.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/File.hpp"

/// \cond
class ComplexDataVector;
/// \endcond

namespace Cce {
/*!
 * \brief The dataset string associated with each scalar that will be output
 * from the `Cce::Tags::worldtube_boundary_tags_for_writing` list (from the tags
 * within the BoundaryPrefix).
 */
template <typename Tag>
std::string dataset_label_for_tag();

/*!
 * \brief Class that standardizes the output of our worldtube data into the
 * Bondi modal format that the CharacteristicExtract executable can read in.
 *
 * \details Takes Bondi nodal data in `fill_data_to_write` and gives the modal
 * data as a `std::vector<double>` with the time as the 0th component. The \p
 * output_l_max that this class is constructed with is the LMax that will be
 * written to disk.
 */
class WorldtubeModeRecorder {
 public:
  WorldtubeModeRecorder();
  WorldtubeModeRecorder(size_t output_l_max, const std::string& h5_filename);

  /// @{
  /*!
   * \brief Writes bondi modal data to the given \p subfile_path
   *
   * \details If nodal data is given (ComplexDataVector), uses `swsh_transform`
   * to transform the data and `libsharp_to_goldberg_modes` to convert the
   * libsharp formatted array into modes.
   *
   * There are exactly half the number of modes for \p Spin = 0 quantities as
   * their are for \p Spin != 0 because we don't include imaginary or m=0 for \p
   * Spin = 0.
   *
   * The \p data_l_max is the LMax of the \p nodal_data or \p modal_data and
   * must be >= \p output_l_max that this class was constructed with. A
   * restriction operation will be performed on the data before it is written,
   * if necessary.
   */
  template <int Spin>
  void append_modal_data(const std::string& subfile_path, double time,
                         const ComplexDataVector& nodal_data,
                         size_t data_l_max);
  template <int Spin>
  void append_modal_data(const std::string& subfile_path, double time,
                         const ComplexModalVector& modal_data,
                         size_t data_l_max);
  /// @}

  /// @{
  /// The legend for writing dat files for both spin = 0 (real) and spin != 0
  /// (all) quantities.
  const std::vector<std::string>& all_legend() const;
  const std::vector<std::string>& real_legend() const;
  /// @}

 private:
  size_t data_to_write_size(bool is_real) const;
  std::vector<std::string> build_legend(bool is_real) const;
  void check_data_l_max(size_t data_l_max) const;

  size_t output_l_max_{};
  h5::H5File<h5::AccessType::ReadWrite> output_file_;
  std::vector<std::string> all_legend_;
  std::vector<std::string> real_legend_;
  std::vector<double> data_to_write_buffer_;
  ComplexModalVector goldberg_mode_buffer_;
};
}  // namespace Cce
