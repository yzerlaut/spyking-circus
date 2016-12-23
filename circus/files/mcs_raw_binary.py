import h5py, numpy, re, sys, re, logging
from circus.shared.messages import print_and_log
from .raw_binary import RawBinaryFile

logger = logging.getLogger(__name__)

class RawMCSFile(RawBinaryFile):

    description    = "mcs_raw_binary"
    extension      = [".raw", ".dat"]

    _required_fields = {'sampling_rate' : float}

    def to_str(self, b, encoding='ascii'):
        """
        Helper function to convert a byte string (or a QByteArray) to a string --
        for Python 3, this specifies an encoding to not end up with "b'...'".
        """
        if sys.version_info[0] == 3:
            return str(b, encoding=encoding)
        else:
            return str(b)

    def _get_header(self):
        try:
            header_size = 0
            stop        = False
            fid         = open(self.file_name, 'rb')
            header_text = ''
            header = {}
            regexp      = re.compile('El_\d*')

            while ((stop is False) and (header_size <= 5000)):
                header_size += 1
                char         = fid.read(1)
                header_text += char.decode('Windows-1252')
                if (header_size > 2):
                    if (header_text[header_size-3:header_size] == 'EOH'):
                        stop = True
            fid.close()
            if stop is False:
                print_and_log(['Wrong MCS header: file is not exported with MCRack'], 'error', logger)
                sys.exit(1)
            else:
                header_size += 2

            f = open(self.file_name, 'rb')
            g = self.to_str(f.read(header_size), encoding='Windows-1252')
            h = g.replace('\r','')
            for i,item in enumerate(h.split('\n')):
                if '=' in item:
                    if item.split(' = ')[0] == 'Di' and len(item.split('=')) == 3:
                        # In case two gains are defined on the same line (digital gain & electrode gain)
                        header['Di'] = item.split(' = ')[1].split(';')[0]
                        header['El'] = item.split(' = ')[2]
                    else:
                        header[item.split(' = ')[0]] = item.split(' = ')[1]
            f.close()

            # Count the number of streams
            streams = header['Streams']
            di_regexp = re.compile("Di_\d*")
            nb_di_streams = len(di_regexp.findall(streams))
            el_regexp = re.compile("El_\d*")
            nb_el_streams = len(el_regexp.findall(streams))
            nb_streams = nb_el_streams

            return header, header_size, nb_streams
        except Exception:
            print_and_log(["Wrong MCS header: file is not exported with MCRack"], 'error', logger)
            sys.exit(1)


    def _read_from_header(self):

        a, b, c                = self._get_header()
        header                 = a
        header['data_offset']  = b
        header['nb_channels']  = c
        header['dtype_offset'] = int(header['ADC zero'])
        header['gain']         = float(re.findall("\d+\.\d+", header['El'])[0])
        header['has_digital']  = 'Di' in header

        if header['dtype_offset'] > 0:
            header['data_dtype'] = 'uint16'
        elif header['dtype_offset'] == 0:
             header['data_dtype'] = 'int16'

        self.data   = numpy.memmap(self.file_name, offset=header['data_offset'], dtype=header['data_dtype'], mode='r')
        self.size   = len(self.data)
        self._shape = (self.size//header['nb_channels'], header['nb_channels'])
        self.has_digital = header['has_digital']
        del self.data

        return header

    def read_chunk(self, idx, chunk_size, padding=(0, 0), nodes=None):

        # Retrieve parameters
        t_start, t_stop = self._get_t_start_t_stop(idx, chunk_size, padding)
        local_shape     = t_stop - t_start
        if self.has_digital:
            nb_channels = self.nb_channels + 1
        else:
            nb_channels = self.nb_channels
        i_start         = t_start * nb_channels
        i_stop          = t_stop * nb_channels

        # Load local chunk
        self._open()
        local_chunk  = self.data[i_start:i_stop]
        local_chunk  = local_chunk.reshape(local_shape, nb_channels)
        self._close()

        if nodes is not None:
            if self.has_digital:
                nodes = nodes + 1
                local_chunk = numpy.take(local_chunk, nodes, axis=1)
            else:
                if not numpy.all(nodes == numpy.arange(nb_channels)):
                    local_chunk = numpy.take(local_chunk, nodes, axis=1)
        else:
            if self.has_digital:
                nodes = numpy.arange(1, nb_channels)
                local_chunk = numpy.take(local_chunk, nodes, axis=1)
                # # TODO clean...
                # print(">>>>>")
                # print("nodes.shape: {}".format(nodes.shape))
                # print("local_chunk.shape: {}".format(local_chunk.shape))
                # print("local_chunk: {}".format(local_chunk))
                # print("<<<<<")
            else:
                pass

        # Scale and cast local chunk
        scaled_casted_local_chunk = self._scale_data_to_float32(local_chunk)
        # # TODO clean...
        # print(">>>>>")
        # print(scaled_casted_local_chunk)
        # import matplotlib.pyplot as plt
        # for i in range(0, 10):
        #     plt.plot(scaled_casted_local_chunk[:, i] + float(50 * i))
        # plt.show()
        # exit()
        # print("<<<<<")

        # Return scaled and casted local chunk
        return scaled_casted_local_chunk
