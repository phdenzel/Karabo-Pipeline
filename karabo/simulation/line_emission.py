import copy
import os
import shutil
from datetime import datetime, timedelta
from typing import Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import oskar
from astropy.constants import c
from astropy.convolution import Gaussian2DKernel
from astropy.io import fits
from astropy.wcs import WCS
from numpy.typing import NDArray

from karabo.imaging.imager import Imager
from karabo.simulation.interferometer import InterferometerSimulation, StationTypeType
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope


def polar_corrdinates_grid(im_shape: Tuple[int, int], center: Tuple[int, int]):
    """
    Creates a corresponding r-phi grid for the x-y coordinate system

    :param im_shape: (x_len, y_len) is the shape of the image in x-y (pixel) coordinates
    :param center: The pixel values of the center (x_center, y_center)
    :return: The corresponding r-phi grid.
    """
    x, y = np.ogrid[: im_shape[0], : im_shape[1]]
    cx, cy = center[0], center[1]

    # convert cartesian --> polar coordinates
    r_array = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    phi_array = np.arctan2(-(x - cx), (y - cy))

    # Needed so that phi = [0, 2*pi] otherwise phi = [-pi, pi]
    phi_array %= 2 * np.pi

    return r_array, phi_array


def circle_image(image: NDArray[np.float_]):
    """
    Cuts the image to a circle, where it takes the x_len/2 as a radius, where x_len is
    the length of the image. Assuming a square image.

    :param image: Input image.
    :return: Image cut, so that only a circle of radius x_len/2 is taken into account.
    """
    x_len, y_len = image.shape
    x_center = x_len / 2 - 1
    y_center = y_len / 2 - 1

    r_array, phi_array = polar_corrdinates_grid((x_len, y_len), (x_center, y_center))

    mask = r_array <= x_center
    image[~mask] = np.NAN

    return image


def header_for_mosaic(img_size: int, ra_deg: float, dec_deg: float, cut: float):
    """
    Create a header for the fits file of the reconstructed image, which is compatible
    with the mosaicking done by MontagePy
    :param img_size: Pixel size of the image.
    :param ra_deg: Right ascension of the center.
    :param dec_deg: Declination of the center.
    :param cut: Size of the reconstructed image in degree.
    :return: Fits header.
    """

    # Create the header
    header = fits.Header()
    header["SIMPLE"] = "T"
    header["BITPIX"] = -64
    header["NAXIS"] = 2
    header["NAXIS1"] = img_size
    header["NAXIS2"] = img_size
    header["CTYPE1"] = "RA---SIN"
    header["CTYPE2"] = "DEC--SIN"
    header["CRVAL1"] = ra_deg
    header["CRVAL2"] = dec_deg
    header["CDELT1"] = -cut / float(img_size)  # 1 arcsecond per pixel
    header["CDELT2"] = cut / float(img_size)
    header["CRPIX1"] = img_size / 2.0
    header["CRPIX2"] = img_size / 2.0
    header["EQUINOX"] = 2000.0

    return header


def rascil_imager(outfile: str, visibility, cut: float = 1.0, img_size: int = 4096):
    """
    Reconstruct the image from the visibilities with rascil.

    :param outfile: Path/Name of the output files.
    :param visibility: Calculated visibilities from sky reconstruction.
    :param cut: Size of the reconstructed image.
    :param img_size: The pixel size of the reconstructed image.
    :return: Dirty image reconstruction of sky.
    """
    cut = cut / 180 * np.pi
    imager = Imager(
        visibility,
        imaging_npixel=img_size,
        imaging_cellsize=cut / img_size,
        imaging_dopsf=True,
    )
    dirty = imager.get_dirty_image()
    dirty.write_to_file(outfile + ".fits", overwrite=True)
    dirty_image = dirty.data[0][0]
    return dirty_image


def oskar_imager(
    outfile: str,
    ra_deg: float = 20,
    dec_deg: float = -30,
    cut: float = 1.0,
    img_size: int = 4096,
):
    """
    Reconstructs the image from the visibilities with oskar.

    :param outfile: Path/Name of the output files.
    :param ra_deg: Phase center right ascension.
    :param dec_deg: Phase center declination.
    :param cut: Size of the reconstructed image.
    :param img_size: The pixel size of the reconstructed image.
    :return: Dirty image reconstruction of sky.
    """
    imager = oskar.Imager()
    # Here plenty of options are available that could be found in the documentation.
    # uv_filter_max can be used to change the baseline length threshold
    imager.set(
        input_file=outfile + ".vis",
        output_root=outfile,
        fov_deg=cut,
        image_size=img_size,
        weighting="Uniform",
        uv_filter_max=3000,
    )
    imager.set_vis_phase_centre(ra_deg, dec_deg)
    imager.output_root = outfile

    output = imager.run(return_images=1)
    image = output["images"][0]
    return image


def plot_scatter_recon(
    sky: SkyModel,
    recon_image: NDArray[np.float_],
    outfile: str,
    header: Optional[fits.header.Header] = None,
    vmin: float = 0,
    vmax: float = 0.4,
    cut: Optional[float] = None,
):
    """
    Plotting the sky as a scatter plot and its reconstruction and saving it as a pdf.

    :param sky: Oskar or Karabo sky.
    :param recon_image: Reconstructed sky from Oskar or Karabo.
    :param outfile: The path of the plot.
    :param header: The header of the recon_image.
    :param vmin: Minimum value of the colorbar.
    :param vmax: Maximum value of the colorbar.
    :param cut: Smaller FOV
    """

    wcs = WCS(header)
    slices = []
    for i in range(wcs.pixel_n_dim):
        if i == 0:
            slices.append("x")
        elif i == 1:
            slices.append("y")
        else:
            slices.append(0)

    # Plot the scatter plot and the sky reconstruction next to each other
    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(121)
    scatter = ax1.scatter(sky[:, 0], sky[:, 1], c=sky[:, 2], vmin=0, s=10, cmap="jet")
    ax1.set_aspect("equal")
    plt.colorbar(scatter, ax=ax1, label="Flux [Jy]")
    if cut is not None:
        ra_deg = header["CRVAL1"]
        dec_deg = header["CRVAL2"]
        ax1.set_xlim((ra_deg - cut / 2, ra_deg + cut / 2))
        ax1.set_ylim((dec_deg - cut / 2, dec_deg + cut / 2))
    ax1.set_xlabel("RA [deg]")
    ax1.set_ylabel("DEC [deg]")
    ax1.invert_xaxis()

    ax2 = fig.add_subplot(122, projection=wcs, slices=slices)
    recon_img = ax2.imshow(
        recon_image, cmap="YlGnBu", origin="lower", vmin=vmin, vmax=vmax
    )
    plt.colorbar(recon_img, ax=ax2, label="Flux Density [Jy]")

    plt.tight_layout()
    plt.savefig(outfile + ".pdf")


def sky_slice(sky: SkyModel, z_obs: NDArray[np.float_], z_min: float, z_max: float):
    """
    Extracting a slice from the sky which includes only sources between redshift z_min
    and z_max.

    :param sky: Sky model.
    :param z_obs: Redshift information of the sky sources. # TODO change as soon as
    branch 400 is merged
    :param z_min: Smallest redshift of this sky bin.
    :param z_max: Largest redshift of this sky bin.

    :return: Sky model only including the sources with redshifts between z_min and
             z_max.
    """
    sky_bin = copy.deepcopy(sky)
    sky_bin_idx = np.where((z_obs > z_min) & (z_obs < z_max))
    sky_bin.sources = sky_bin.sources[sky_bin_idx]

    return sky_bin


def redshift_slices(redshift_obs: NDArray[np.float_], channel_num: int = 10):
    print("Smallest redshift:", np.amin(redshift_obs))
    print("Largest redshift:", np.amax(redshift_obs))

    redshift_channel = np.linspace(
        np.amin(redshift_obs), np.amax(redshift_obs), channel_num + 1
    )

    return redshift_channel


def freq_channels(z_obs: NDArray[np.float_], channel_num: int = 10):
    """
    Calculates the frequency channels from the redshifs.
    :param z_obs: Observed redshifts from the HI sources.
    :param channel_num: Number uf channels.

    :return: Redshift channel, frequency channel in Hz, bin width of frequency channel
             in Hz, middle frequency in Hz
    """

    redshift_channel = redshift_slices(z_obs, channel_num)

    freq_channel = c.value / (0.21 * (1 + redshift_channel))
    freq_start = freq_channel[0]
    freq_end = freq_channel[-1]
    freq_mid = freq_start + (freq_end - freq_start) / 2
    freq_bin = freq_channel[0] - freq_channel[1]
    print("The frequency channel starts at:", freq_start, "Hz")
    print("The bin size of the freq channel is:", freq_bin, "Hz")
    print("The freq channel: ", freq_channel)

    return redshift_channel, freq_channel, freq_bin, freq_mid


def karabo_reconstruction(
    outfile: str,
    mosaic_pntg_file: Optional[str] = None,
    sky: Optional[SkyModel] = None,
    ra_deg: float = 20,
    dec_deg: float = -30,
    start_time=datetime(2000, 3, 20, 12, 6, 39),
    obs_length=timedelta(hours=3, minutes=5, seconds=0, milliseconds=0),
    start_freq: float = 1.4639e9,
    freq_bin: float = 1.0e7,
    beam_type: StationTypeType = "Isotropic beam",
    gaussian_fwhm: float = 1.0,
    gaussian_ref_freq: float = 1.4639e9,
    cut: float = 1.0,
    img_size: int = 4096,
    channel_num: int = 10,
    pdf_plot: bool = False,
    circle: bool = False,
    rascil: bool = True,
):
    """
    Performs a sky reconstruction for our test sky.

    :param outfile: Path/Name of the output files.
    :param mosaic_pntg_file: If provided an additional output fits file which has the
                             correct format for creating a
                             mosaic with Montage is created and saved at this path.
    :param sky: Sky model. If None, a test sky (out of equally spaced sources) is used.
    :param ra_deg: Phase center right ascension.
    :param dec_deg: Phase center declination.
    :param start_time: Observation start time.
    :param obs_length: Observation length (time).
    :param start_freq: The frequency at the midpoint of the first channel in Hz.
    :param freq_bin: The frequency width of the channel.
    :param beam_type: Primary beam assumed, e.g. "Isotropic beam", "Gaussian beam",
                      "Aperture Array".
    :param gaussian_fwhm: If the primary beam is gaussian, this is its FWHM. In power
                          pattern. Units = degrees.
    :param gaussian_ref_freq: If you choose "Gaussian beam" as station type you need
                              specify the reference frequency of
                              the reference frequency of the full-width half maximum
                              here.
    :param cut: Size of the reconstructed image.
    :param img_size: The pixel size of the reconstructed image.
    :param channel_num:
    :param pdf_plot: Shall we plot the scatter plot and the reconstruction as a pdf?
    :param circle: If set to True, the pointing has a round shape of size cut.
    :param rascil: If True we use the Imager Rascil otherwise the Imager from Oskar is
                   used.
    :return: Reconstructed sky of one pointing of size cut.
    """
    print("Create Sky...")
    if sky is None:
        sky = SkyModel.sky_test()

    telescope = Telescope.get_MEERKAT_Telescope()

    print("Sky Simulation...")
    simulation = InterferometerSimulation(
        vis_path=outfile + ".vis",
        channel_bandwidth_hz=1.0e7,
        time_average_sec=8,
        ignore_w_components=True,
        uv_filter_max=3000,
        use_gpus=True,
        station_type=beam_type,
        enable_power_pattern=True,
        gauss_beam_fwhm_deg=gaussian_fwhm,
        gauss_ref_freq_hz=gaussian_ref_freq,
    )
    print("Setup observation parameters...")
    observation = Observation(
        phase_centre_ra_deg=ra_deg,
        phase_centre_dec_deg=dec_deg,
        start_date_and_time=start_time,
        length=obs_length,
        number_of_time_steps=10,
        start_frequency_hz=start_freq,
        frequency_increment_hz=freq_bin,
        number_of_channels=channel_num,
    )
    print("Calculate visibilites...")
    visibility = simulation.run_simulation(telescope, sky, observation)

    if rascil:
        print("Sky reconstruction with Rascil...")
        dirty_image = rascil_imager(outfile, visibility, cut, img_size)
    else:
        print("Sky reconstruction with the Oskar Imager")
        dirty_image = oskar_imager(outfile, ra_deg, dec_deg, cut, img_size)

    if circle:
        print("Cutout a circle from image...")
        dirty_image = circle_image(dirty_image)

    header = header_for_mosaic(img_size, ra_deg, dec_deg, cut)
    if pdf_plot:
        print(
            "Creation of a pdf with scatter plot and reconstructed image to ",
            str(outfile),
        )
        plot_scatter_recon(sky, dirty_image, outfile, header)

    if mosaic_pntg_file is not None:
        print(
            "Write the reconstructed image to a fits file which can be used for "
            "coaddition.",
            mosaic_pntg_file,
        )
        fits.writeto(mosaic_pntg_file + ".fits", dirty_image, header, overwrite=True)

    return dirty_image, header


def line_emission_pointing(
    path_outfile: str,
    sky: SkyModel,
    z_obs: NDArray[np.float_],  # TODO: After branch 400-read_in_sky-exists the sky
    # includes this information -> rewrite
    ra_deg: float = 20,
    dec_deg: float = -30,
    num_bins: int = 10,
    beam_type: StationTypeType = "Gaussian beam",
    gaussian_fwhm: float = 1.0,
    gaussian_ref_freq: float = 1.4639e9,
    start_time=datetime(2000, 3, 20, 12, 6, 39),
    obs_length=timedelta(hours=3, minutes=5, seconds=0, milliseconds=0),
    cut: float = 3.0,
    img_size: int = 4096,
    circle: bool = True,
    rascil: bool = True,
):
    """
    Simulating line emission for one pointing.

    :param path_outfile: Pathname of the output file and folder.
    :param sky: Sky model which is used for simulating line emission. If None, a test
                sky (out of equally spaced sources) is used.
    :param z_obs: Redshift information of the sky sources.
    :param ra_deg: Phase center right ascension.
    :param dec_deg: Phase center declination.
    :param num_bins: Number of redshift/frequency slices used to simulate line emission.
                     The more the better the line emission is simulated.
    :param beam_type: Primary beam assumed, e.g. "Isotropic beam", "Gaussian beam",
                      "Aperture Array".
    :param gaussian_fwhm: If the primary beam is gaussian, this is its FWHM. In power
                          pattern. Units = degrees.
    :param gaussian_ref_freq: If you choose "Gaussian beam" as station type you need
                              specify the reference frequency of the reference
                              frequency of the full-width half maximum here.
    :param start_time: Observation start time.
    :param obs_length: Observation length (time).
    :param cut: Size of the reconstructed image.
    :param img_size: The pixel size of the reconstructed image.
    :param circle: If set to True, the pointing has a round shape of size cut.
    :param rascil: If True we use the Imager Rascil otherwise the Imager from Oskar is
                   used.
    :return: Total line emission reconstruction, 3D line emission reconstruction,
             Header of reconstruction and mean frequency.


    E.g. for how to do the simulation of line emission for one pointing and then
    applying gaussian primary beam correction to it.

    outpath = (
        "/home/user/Documents/SKAHIIM_Pipeline/result/Reconstructions/"
        "Line_emission_pointing_2"
    )
    catalog_path = (
        "/home/user/Documents/SKAHIIM_Pipeline/Flux_calculation/"
        "Catalog/point_sources_OSKAR1_FluxBattye_diluted5000.h5"
    )
    ra = 20
    dec = -30
    sky_pointing, z_obs_pointing = SkyModel.sky_from_h5_with_redshift(
        catalog_path, ra, dec
    )
    dirty_im, _, header_dirty, freq_mid_dirty = line_emission_pointing(
        outpath, sky_pointing, z_obs_pointing
    )
    plot_scatter_recon(
        sky_pointing, dirty_im, outpath, header_dirty, vmax=0.15, cut=3.0
    )
    gauss_fwhm = gaussian_fwhm_meerkat(freq_mid_dirty)
    beam_corrected, _ = simple_gaussian_beam_correction(outpath, dirty_im, gauss_fwhm)
    plot_scatter_recon(
        sky_pointing,
        beam_corrected,
        outpath + "_GaussianBeam_Corrected",
        header_dirty,
        vmax=0.15,
        cut=3.0,
    )
    """
    # Create folders to save outputs/ delete old one if it already exists
    try:
        shutil.rmtree(path_outfile)
    except FileNotFoundError:
        print(
            "                Can't delete work tree; probably doesn't exist yet",
            flush=True,
        )

    print("Work directory: " + path_outfile, flush=True)
    os.makedirs(path_outfile)

    redshift_channel, freq_channel, freq_bin, freq_mid = freq_channels(z_obs, num_bins)

    dirty_images = []
    header = None

    for bin_idx in range(num_bins):
        print("Channel " + str(bin_idx) + " is being processed...")

        print("Extracting the corresponding frequency slice from the sky model...")
        sky_bin = sky_slice(
            sky, z_obs, redshift_channel[bin_idx], redshift_channel[bin_idx + 1]
        )

        print("Starting simulation...")
        start_freq = freq_channel[bin_idx] + freq_bin / 2
        dirty_image, header = karabo_reconstruction(
            path_outfile + "/slice_" + str(bin_idx),
            sky=sky_bin,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            start_freq=start_freq,
            freq_bin=freq_bin,
            beam_type=beam_type,
            gaussian_fwhm=gaussian_fwhm,
            gaussian_ref_freq=gaussian_ref_freq,
            start_time=start_time,
            obs_length=obs_length,
            cut=cut,
            img_size=img_size,
            channel_num=1,
            circle=circle,
            rascil=rascil,
        )

        dirty_images.append(dirty_image)

    dirty_image = sum(dirty_images)

    print("Save summed dirty images as fits file")
    dirty_img = fits.PrimaryHDU(dirty_image)
    dirty_img.writeto(path_outfile + ".fits", overwrite=True)

    print("Save 3-dim reconstructed dirty images as h5")
    z_bin = redshift_channel[1] - redshift_channel[0]
    z_channel_mid = redshift_channel + z_bin / 2

    f = h5py.File(path_outfile + ".h5", "w")
    dataset_dirty = f.create_dataset("Dirty Images", data=dirty_images)
    dataset_dirty.attrs["Units"] = "Jy"
    f.create_dataset("Observed Redshift Channel Center", data=z_channel_mid)
    f.create_dataset("Observed Redshift Bin Size", data=z_bin)

    return dirty_image, dirty_images, header, freq_mid


def gaussian_fwhm_meerkat(freq: float):
    """
    Computes the FWHM of MeerKAT for a certain observation frequency.

    :param freq: Frequency of interest in Hz.
    :return: The power pattern FWHM of the MeerKAT telescope at this frequency in
             degrees.
    """
    gaussian_fwhm = np.sqrt(89.5 * 86.2) / 60.0 * (1e3 / (freq / 10**6))

    return gaussian_fwhm


def gaussian_beam(
    ra_deg: float,
    dec_deg: float,
    img_size: int = 2048,
    cut: float = 1.2,
    fwhm: float = 1.0,
    outfile: str = "beam",
):
    """
    Creates a Gaussian beam at RA, DEC.
    :param ra_deg: Right ascension coordinate of center of Gaussian.
    :param dec_deg: Declination coordinate of center of Gaussian.
    :param img_size: Pixel image size.
    :param cut: Image size in degrees.
    :param fwhm: FWHM of the Gaussian in degrees.
    :param outfile: Name of the image file with the Gaussian.
    :return:
    """
    # We create the image header and the wcs frame of the Gaussian
    header = header_for_mosaic(
        img_size=img_size, ra_deg=ra_deg, dec_deg=dec_deg, cut=cut
    )
    wcs = WCS(header)

    # Calculate Gaussian
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    gauss_kernel = Gaussian2DKernel(
        sigma / wcs.wcs.cdelt[0], x_size=img_size, y_size=img_size
    )

    beam_image = gauss_kernel.array
    # normalize the kernel, such that max=1.0
    beam_image = beam_image / np.max(beam_image)

    # make the beam image circular and save it as a fits file
    beam_image = circle_image(beam_image)
    fits.writeto(outfile + ".fits", beam_image, header, overwrite=True)

    return beam_image, header


def simple_gaussian_beam_correction(
    path_outfile: str,
    dirty_image: NDArray[np.float_],
    gaussian_fwhm: float,
    ra_deg: float = 20,
    dec_deg: float = -30,
    cut: float = 3.0,
    img_size: int = 4096,
):
    print("Calculate gaussian beam for primary beam correction...")
    beam, header = gaussian_beam(
        img_size=img_size,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        cut=cut,
        fwhm=gaussian_fwhm,
        outfile=path_outfile + "/gaussian_beam",
    )

    print("Apply primary beam correction...")
    dirty_image_corrected = dirty_image / beam

    fits.writeto(
        path_outfile + "_GaussianBeam_Corrected.fits",
        dirty_image_corrected,
        header,
        overwrite=True,
    )

    return dirty_image_corrected, header
